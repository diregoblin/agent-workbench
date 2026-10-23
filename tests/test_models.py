import contextlib
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from agent_workbench import models


class Handler(BaseHTTPRequestHandler):
    mode = 'ok'
    calls = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.respond()

    def do_POST(self):
        self.respond()

    def respond(self):
        raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        type(self).calls.append((self.path, json.loads(raw) if raw else None, self.headers.get('Authorization')))
        if self.mode == 'slow':
            time.sleep(1)
        if self.mode == 'redirect':
            self.send_response(302)
            self.send_header('Location', 'http://127.0.0.1:1/never')
            self.end_headers()
            return
        self.send_response(401 if self.mode == 'error' else 200)
        self.end_headers()
        data = {'data': [{'id': 'fixture'}]}
        if self.path.endswith('/chat/completions'):
            data = {'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'length'}]}
        if self.path.endswith('/embeddings'):
            data = {'data': [{'embedding': [0.1, 0.2]}]}
        raw = json.dumps(data).encode()
        if self.mode == 'invalid':
            raw = b'{"error":"secret text"}'
        if self.mode == 'large':
            raw = b'x' * (models.MAX_BYTES + 1)
        if self.mode == 'partial':
            raw = b'{"data":'
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Handler.mode, Handler.calls = 'ok', []

    def test_catalog_not_inference_and_no_retry(self):
        result = models.openai_request(self.url, 'health')
        self.assertEqual(result['models'], ['fixture'])
        self.assertFalse(result['evidence']['inference'])
        self.assertEqual(len(Handler.calls), 1)

    def test_chat_embedding_explicit_model(self):
        with patch.dict(os.environ, {'TEST_TOKEN': 'fixture-key'}):
            result = models.openai_request(self.url+'/v1/', 'chat', model='chosen', text='private', key_env='TEST_TOKEN', max_tokens=7)
        self.assertTrue(result['truncated'])
        self.assertEqual(Handler.calls[-1][1]['model'], 'chosen')
        self.assertEqual(Handler.calls[-1][1]['max_tokens'], 7)
        self.assertEqual(Handler.calls[-1][2], 'Bearer fixture-key')
        result = models.openai_request(self.url, 'embed', model='chosen', text='private')
        self.assertEqual(result['dimensions'], 2)

    def test_errors_and_partial_output(self):
        for mode, code in [('redirect', 'http_302'), ('error', 'http_401'), ('partial', 'transport_or_json_error'), ('large', 'response_too_large'), ('invalid', 'invalid_catalog')]:
            Handler.mode = mode
            with self.subTest(mode=mode), self.assertRaises(models.ModelError) as ctx:
                models.openai_request(self.url, 'list')
            self.assertEqual(ctx.exception.code, code)

    def test_timeout_and_cancel(self):
        Handler.mode = 'slow'
        start = time.monotonic()
        with self.assertRaises(models.ModelError) as ctx:
            models.openai_request(self.url, 'list', timeout=.2)
        self.assertEqual(ctx.exception.status, 4)
        self.assertLess(time.monotonic()-start, 1)
        event = threading.Event()
        timer = threading.Timer(.2, event.set)
        timer.start()
        try:
            with self.assertRaises(models.ModelError) as ctx:
                models.openai_request(self.url, 'list', cancel=event)
            self.assertEqual(ctx.exception.status, 130)
        finally:
            timer.join()

    def test_invalid_inputs_and_redacted_dry_run(self):
        for url in ['https://example.org', 'http://127.0.0.1@evil.test', 'http://localhost/?key=x', 'http://localhost:bad', 'file:///tmp/a']:
            with self.subTest(url=url), self.assertRaises(models.ModelError):
                models.openai_request(url, 'list', dry_run=True)
        result = models.openai_request(self.url, 'chat', model='chosen', text='secret', key_env='MISSING', dry_run=True)
        self.assertNotIn('secret', json.dumps(result))
        self.assertEqual(Handler.calls, [])
        for timeout in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(models.ModelError):
                models.openai_request(self.url, 'list', timeout=timeout)


FIXTURE = '''#!/usr/bin/env python3
import json, os, pathlib, sys, time, wave
args=sys.argv[1:]
mode=os.environ.get('FIXTURE_MODE','ok')
if mode=='slow': time.sleep(10)
if mode=='fail':
 print('private text and key',file=sys.stderr)
 sys.exit(7)
if '--no-play' in args:
 assert args[args.index('--device')+1]=='cpu'
 assert os.environ['OMNIVOICE_MODEL'].endswith('large-v3-turbo')
 assert args[-2]=='--'
 out=pathlib.Path(args[args.index('--output')+1])
 with wave.open(str(out),'wb') as w:
  w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b'\\x00\\x00'*240)
else:
 assert args[args.index('--model')+1]=='large-v3-turbo'
 assert args[args.index('--compute_type')+1]=='int8'
 assert args[args.index('--local_files_only')+1]=='True'
 out=pathlib.Path(args[args.index('--output_dir')+1])/(pathlib.Path(args[0]).stem+'.json')
 out.write_text(json.dumps({'text':'fixture transcript'}))
if mode=='partial': out.write_bytes(b'bad partial')
'''


@unittest.skipUnless(os.name == 'posix', 'POSIX process group execution')
class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.launcher = self.root/'launcher'
        self.launcher.write_text(FIXTURE)
        self.launcher.chmod(0o700)
        self.model = self.root/'large-v3-turbo'
        self.model.mkdir()
        (self.model/'model.bin').touch()
        self.audio = self.root/'audio ; $(touch injected).wav'
        self.audio.touch()
        self.out = self.root/'output'
        self.kwargs = dict(launcher=str(self.launcher), model_path=str(self.model), output_dir=str(self.out))

    def tearDown(self):
        self.temp.cleanup()

    def test_whisper_safe_argv_artifact(self):
        result = models.speech('whisper', audio=str(self.audio), **self.kwargs)
        self.assertEqual(json.loads(Path(result['artifact']).read_text())['text'], 'fixture transcript')
        self.assertTrue(result['evidence']['inference'])
        self.assertFalse((self.root/'injected').exists())
        with self.assertRaises(models.ModelError):
            models.speech('whisper', audio=str(self.audio), **self.kwargs)

    def test_tts_no_play_and_literal_text(self):
        result = models.speech('tts', text='--device cuda ; $(touch injected)', **self.kwargs)
        self.assertFalse(result['playback'])
        self.assertTrue(Path(result['artifact']).is_file())

    def test_dry_run_no_process_or_output(self):
        with patch.object(models.subprocess, 'Popen', side_effect=AssertionError):
            result = models.speech('tts', text='private', dry_run=True, **self.kwargs)
        self.assertNotIn('private', json.dumps(result))
        self.assertFalse(self.out.exists())

    def test_failure_partial_timeout_cleanup(self):
        for operation in ['tts', 'whisper']:
            for mode in ['fail', 'partial', 'slow']:
                with self.subTest(operation=operation, mode=mode), patch.dict(os.environ, {'FIXTURE_MODE': mode}):
                    with self.assertRaises(models.ModelError):
                        models.speech(operation, text='private', audio=str(self.audio), timeout=.15 if mode=='slow' else 2, **self.kwargs)
                self.assertFalse(self.out.exists())
                self.assertEqual(list(self.root.glob('.model-*')), [])

    @unittest.skipUnless(Path('/proc').is_dir(), 'Linux descendant state check')
    def test_timeout_kills_descendant_group(self):
        pidfile = self.root/'descendant.pid'
        self.launcher.write_text('#!/usr/bin/env python3\nimport subprocess,time,pathlib\np=subprocess.Popen(["sleep","30"])\npathlib.Path('+repr(str(pidfile))+').write_text(str(p.pid))\ntime.sleep(30)\n')
        with self.assertRaises(models.ModelError) as ctx:
            models.speech('tts', text='fixture', timeout=.4, **self.kwargs)
        self.assertEqual(ctx.exception.status, 4)
        self.assertTrue(pidfile.exists())
        stat = Path('/proc')/pidfile.read_text()/'stat'
        deadline = time.monotonic()+1
        while stat.exists() and stat.read_text().split()[2] != 'Z' and time.monotonic()<deadline:
            time.sleep(.02)
        if stat.exists():
            self.assertEqual(stat.read_text().split()[2], 'Z')
        self.assertFalse(self.out.exists())

    def test_cancel_and_wrong_model(self):
        event = threading.Event()
        timer = threading.Timer(.1, event.set)
        timer.start()
        try:
            with patch.dict(os.environ, {'FIXTURE_MODE': 'slow'}), self.assertRaises(models.ModelError) as ctx:
                models.speech('tts', text='private', cancel=event, **self.kwargs)
            self.assertEqual(ctx.exception.status, 130)
        finally:
            timer.join()
        self.model.rename(self.root/'small')
        self.kwargs['model_path'] = str(self.root/'small')
        with self.assertRaises(models.ModelError):
            models.speech('whisper', audio=str(self.audio), **self.kwargs)


class CliTests(unittest.TestCase):
    def test_json_error_without_argument_leak(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = models.main(['chat', '--private-secret'])
        self.assertEqual(status, 2)
        self.assertFalse(json.loads(stdout.getvalue())['ok'])
        self.assertNotIn('private-secret', stderr.getvalue())

    def test_sigterm_cleans_running_http_worker(self):
        import subprocess
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        Handler.mode, Handler.calls = 'slow', []
        process = subprocess.Popen([os.sys.executable, '-m', 'agent_workbench.models', 'health', '--endpoint', 'http://127.0.0.1:'+str(server.server_port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic()+3
            while not Handler.calls and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertTrue(Handler.calls)
            process.terminate()
            stdout, stderr = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 130)
            self.assertEqual(json.loads(stdout)['error'], 'cancelled')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            server.shutdown()
            server.server_close()
            thread.join()
            Handler.mode = 'ok'

    def test_module_entrypoint(self):
        import subprocess
        result = subprocess.run([os.sys.executable, '-m', 'agent_workbench.models', 'list', '--endpoint', 'http://127.0.0.1:1', '--dry-run'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(json.loads(result.stdout)['dry_run'])


if __name__ == '__main__':
    unittest.main()
