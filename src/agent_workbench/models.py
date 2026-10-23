"""Explicit, bounded local model adapters; no import-time probes or downloads."""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

MAX_BYTES = 4 * 1024 * 1024


class ModelError(Exception):
    def __init__(self, code: str, status: int = 3):
        super().__init__(code)
        self.code, self.status = code, status


def _timeout(value):
    if not math.isfinite(value) or not 0 < value <= 3600:
        raise ModelError('timeout_must_be_between_0_and_3600', 2)


def _cancelled(cancel):
    return cancel is not None and cancel.is_set()


def _endpoint(base):
    p = urllib.parse.urlsplit(base)
    try:
        local = p.hostname == 'localhost' or ipaddress.ip_address(p.hostname).is_loopback
    except ValueError:
        local = False
    if (p.scheme not in ('http', 'https') or not local or p.username or p.password
            or p.query or p.fragment or p.path.rstrip('/') not in ('', '/v1')):
        raise ModelError('endpoint_requires_loopback_url_without_credentials_or_query', 2)
    try:
        p.port
    except ValueError:
        raise ModelError('invalid_endpoint_port', 2) from None
    return base.rstrip('/') + ('' if p.path.rstrip('/') == '/v1' else '/v1')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_worker(conn, url, payload, key, timeout):
    # Separate process bounds even slow headers/bodies. No proxy or redirect escape.
    try:
        headers = {'Accept': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(url, data=data, headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ModelError('response_too_large')
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ModelError('invalid_response')
            conn.send((True, result))
    except urllib.error.HTTPError as exc:
        conn.send((False, 'http_' + str(exc.code)))
    except ModelError as exc:
        conn.send((False, exc.code))
    except Exception:
        conn.send((False, 'transport_or_json_error'))
    finally:
        conn.close()


def openai_request(endpoint: str, operation: str, *, model: str | None = None,
                   text: str | None = None, key_env: str | None = None,
                   max_tokens: int = 128, timeout: float = 30,
                   dry_run: bool = False, cancel=None) -> dict:
    """GET catalog/health or one POST. Never chooses a model or retries."""
    _timeout(timeout)
    base = _endpoint(endpoint)
    if operation not in ('health', 'list', 'chat', 'embed'):
        raise ModelError('unknown_operation', 2)
    inference = operation in ('chat', 'embed')
    if inference and (not model or not text or len(text.encode()) > 65536):
        raise ModelError('explicit_model_and_1_to_65536_byte_text_required', 2)
    if not 1 <= max_tokens <= 4096:
        raise ModelError('max_tokens_must_be_1_to_4096', 2)
    path, payload = '/models', None
    if operation == 'chat':
        path = '/chat/completions'
        payload = dict(model=model, messages=[dict(role='user', content=text)],
                       max_tokens=max_tokens, stream=False)
    elif operation == 'embed':
        path, payload = '/embeddings', dict(model=model, input=text)
    evidence = dict(configured=True, http_response=False, catalog=False, inference=False)
    if dry_run:
        return dict(ok=True, dry_run=True, operation=operation, endpoint=base,
                    model=model, method='POST' if inference else 'GET', path=path,
                    evidence=evidence)
    key = os.environ.get(key_env) if key_env else None
    if key_env and not key:
        raise ModelError('credential_environment_variable_missing', 2)
    if _cancelled(cancel):
        raise ModelError('cancelled', 130)
    ctx = multiprocessing.get_context('spawn')
    receive, send = ctx.Pipe(duplex=False)
    worker = ctx.Process(target=_http_worker, args=(send, base + path, payload, key, timeout))
    worker.start()
    send.close()
    deadline = time.monotonic() + timeout
    try:
        while not receive.poll(min(.05, max(0, deadline - time.monotonic()))):
            if _cancelled(cancel):
                raise ModelError('cancelled', 130)
            if time.monotonic() >= deadline:
                raise ModelError('timeout', 4)
            if not worker.is_alive():
                raise ModelError('http_worker_failed')
        try:
            ok, result = receive.recv()
        except EOFError:
            raise ModelError('http_worker_failed') from None
        if not ok:
            raise ModelError(result)
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=1)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=1)
        receive.close()
    evidence['http_response'] = True
    if operation in ('list', 'health'):
        rows = result.get('data')
        if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get('id'), str) for r in rows):
            raise ModelError('invalid_catalog')
        evidence['catalog'] = True
        return dict(ok=True, evidence=evidence, models=[r['id'] for r in rows])
    if operation == 'chat':
        try:
            content = result['choices'][0]['message']['content']
            finish = result['choices'][0]['finish_reason']
            if not isinstance(content, str) or finish not in ('stop', 'length'):
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelError('invalid_chat_response') from None
        output = dict(text=content, finish_reason=finish, truncated=finish == 'length')
    else:
        try:
            vector = result['data'][0]['embedding']
            if not isinstance(vector, list) or not vector or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelError('invalid_embedding_response') from None
        output = dict(embedding=vector, dimensions=len(vector))
    evidence['inference'] = True
    return dict(ok=True, model=model, evidence=evidence, **output)


def _path(value, kind):
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if kind == 'directory' else path.is_file()):
        raise ModelError('required_' + kind + '_missing', 2)
    return path


def _run(argv, timeout, cancel, extra_env=None):
    if os.name != 'posix':
        raise ModelError('speech_execution_requires_posix_process_groups', 2)
    if _cancelled(cancel):
        raise ModelError('cancelled', 130)
    # Child logs can contain prompts/transcripts/credentials. Deliberately discard.
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.update(extra_env or {})
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True, env=env)
    deadline = time.monotonic() + timeout
    try:
        while child.poll() is None:
            if _cancelled(cancel):
                raise ModelError('cancelled', 130)
            if time.monotonic() >= deadline:
                raise ModelError('timeout', 4)
            time.sleep(.02)
        if child.returncode:
            raise ModelError('launcher_failed')
    finally:
        # Kill the group even if the leader exited with lingering descendants.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=2)


def speech(operation: str, *, launcher: str, model_path: str, output_dir: str,
           audio: str | None = None, text: str | None = None,
           device: str = 'cpu', timeout: float = 300,
           dry_run: bool = False, cancel=None) -> dict:
    """Publish a new artifact directory only after success; failure removes staging."""
    _timeout(timeout)
    if operation not in ('whisper', 'tts') or device not in ('cpu', 'cuda'):
        raise ModelError('invalid_speech_operation_or_device', 2)
    executable = _path(launcher, 'file')
    if not os.access(executable, os.X_OK):
        raise ModelError('launcher_not_executable', 2)
    model = _path(model_path, 'directory')
    if operation == 'whisper':
        if model.name != 'large-v3-turbo' or not (model / 'model.bin').is_file():
            raise ModelError('local_large_v3_turbo_model_required', 2)
        source = _path(audio, 'file') if audio else None
        if source is None:
            raise ModelError('audio_file_required', 2)
    elif not text or len(text.encode()) > 16384:
        raise ModelError('1_to_16384_byte_text_required', 2)
    target = Path(output_dir).expanduser().absolute()
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise ModelError('output_directory_must_be_new_with_existing_parent', 2)
    evidence = dict(configured=True, files=True, inference=False)
    if dry_run:
        return dict(ok=True, dry_run=True, operation=operation, device=device,
                    launcher=str(executable), model_path=str(model), output_dir=str(target),
                    playback=False, evidence=evidence)
    # Reserve target against concurrent writers. Partial data stays in private staging.
    target.mkdir(mode=0o700)
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix='.model-', dir=target.parent))
        if operation == 'whisper':
            argv = [str(executable), str(source), '--model', 'large-v3-turbo',
                    '--model_directory', str(model), '--local_files_only', 'True',
                    '--device', device, '--compute_type', 'int8' if device == 'cpu' else 'int8_float16',
                    '--output_dir', str(stage), '--output_format', 'json']
            artifact = stage / (source.stem + '.json')
            _run(argv, timeout, cancel)
            if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size > MAX_BYTES:
                raise ModelError('missing_or_oversized_artifact')
            try:
                data = json.loads(artifact.read_text())
                if not isinstance(data, dict) or not isinstance(data.get('text'), str):
                    raise ValueError()
            except (ValueError, UnicodeError):
                raise ModelError('invalid_transcript_artifact') from None
        else:
            artifact = stage / 'speech.wav'
            # Explicit model env only for child, without modifying parent environment.
            argv = [str(executable), '--device', 'cpu' if device == 'cpu' else 'cuda:0',
                    '--output', str(artifact), '--no-play', '--', text]
            _run_tts(argv, timeout, cancel, model)
            if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size > 64 * 1024 * 1024:
                raise ModelError('missing_or_oversized_audio_artifact')
            try:
                with wave.open(str(artifact), 'rb') as wav:
                    expected = wav.getnframes() * wav.getnchannels() * wav.getsampwidth()
                    actual = 0
                    while chunk := wav.readframes(65536):
                        actual += len(chunk)
                    if expected <= 0 or actual != expected:
                        raise ValueError()
            except (wave.Error, EOFError, ValueError):
                raise ModelError('invalid_audio_artifact') from None
        destination = target / artifact.name
        artifact.replace(destination)
        evidence['inference'] = True
        return dict(ok=True, evidence=evidence, artifact=str(destination),
                    bytes=destination.stat().st_size, device=device, playback=False)
    except BaseException:
        shutil.rmtree(target)
        raise
    finally:
        if stage is not None:
            shutil.rmtree(stage)


def _run_tts(argv, timeout, cancel, model):
    _run(argv, timeout, cancel, extra_env={'OMNIVOICE_MODEL': str(model)})


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ModelError('invalid_arguments_use_help', 2)


def _main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    for name in ('health', 'list', 'chat', 'embed', 'whisper', 'tts'):
        p = commands.add_parser(name)
        p.add_argument('--timeout', type=float, default=30 if name in ('health', 'list', 'chat', 'embed') else 300)
        p.add_argument('--dry-run', action='store_true')
        if name in ('health', 'list', 'chat', 'embed'):
            p.add_argument('--endpoint', required=True)
            p.add_argument('--key-env')
            if name in ('chat', 'embed'):
                p.add_argument('--model', required=True)
                p.add_argument('--text-file', required=True, help='UTF-8 file, or - for stdin')
                p.add_argument('--max-tokens', type=int, default=128)
        else:
            p.add_argument('--launcher', required=True)
            p.add_argument('--model-path', required=True)
            p.add_argument('--output-dir', required=True)
            p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
            p.add_argument('--audio', required=True) if name == 'whisper' else p.add_argument('--text-file', required=True)
    try:
        args = vars(parser.parse_args(argv))
        operation = args.pop('operation')
        file = args.pop('text_file', None)
        if file:
            if file == '-':
                raw = sys.stdin.buffer.read(65537)
            else:
                with open(file, 'rb') as handle:
                    raw = handle.read(65537)
            if len(raw) > 65536:
                raise ModelError('text_too_large', 2)
            args['text'] = raw.decode('utf-8')
        result = (openai_request(operation=operation, **args) if operation in ('health', 'list', 'chat', 'embed')
                  else speech(operation, **args))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        error = ModelError('cancelled', 130)
    except ModelError as exc:
        error = exc
    except (OSError, ValueError, TypeError):
        error = ModelError('local_io_or_configuration_error', 2)
    print(json.dumps(dict(ok=False, error=error.code)))
    print(error.code, file=sys.stderr)
    return error.status


def main(argv: list[str] | None = None) -> int:
    # Restore the integrating caller's handler after this invocation.
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return _main(argv)
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
