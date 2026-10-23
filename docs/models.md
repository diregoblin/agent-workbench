# Local auxiliary models

`python -m agent_workbench.models` exposes small, stdlib-only adapters, not a model
server or scheduler. Python callers can use `openai_request`, `speech`, or
`main(argv: list[str] | None = None) -> int`. Imports have no operational effects.

## Existing solutions first

The maintained [OpenAI Python SDK](https://github.com/openai/openai-python) already
covers the standard API; use it if you need streaming, tools, or remote providers.
This module deliberately wraps only the small OpenAI-compatible HTTP subset using
stdlib, with no dependency, implicit endpoint, fallback, or retries.
[ faster-whisper ](https://github.com/SYSTRAN/faster-whisper) supplies ASR; we invoke
an existing whisper-ctranslate2-compatible launcher, not a new ASR implementation.
TTS invokes an existing OmniVoice-compatible launcher. Nothing installs weights,
changes services, chooses another model, or downloads dependencies.
The two repositories were checked as non-archived on 2026-10-23.

## HTTP

```sh
PYTHONPATH=src python -m agent_workbench.models health --endpoint http://127.0.0.1:8080
PYTHONPATH=src python -m agent_workbench.models list --endpoint http://127.0.0.1:8080/v1
PYTHONPATH=src python -m agent_workbench.models chat --endpoint http://127.0.0.1:8080/v1 --model YOUR_MODEL_ID --text-file prompt.txt --max-tokens 128 --timeout 30
PYTHONPATH=src python -m agent_workbench.models embed --endpoint http://127.0.0.1:8080/v1 --model YOUR_EMBEDDING_ID --text-file input.txt --dry-run
```

Only loopback IPs and `localhost` are accepted. Credentials, query strings and
fragments in URLs are rejected. HTTP proxies and all redirects are disabled.
Optional `--key-env LOCAL_MODEL_API_KEY` reads a credential only from that named
environment variable; the value is not printed or written. No config file parser
is provided: the example TOML documents settings for a caller to map to arguments.
Use `--text-file -` for stdin; prompts are never echoed in plans/errors. Successful
chat output and embedding vectors **are returned intentionally** on JSON stdout.
Consumers should handle that output as potentially private.

Health is GET `/v1/models`, not inference readiness. The result distinguishes
`configured`, `http_response`, `catalog`, and `inference`. A model name in the
catalog does not establish that loading or generation succeeds. Chat and embed
require an explicit model and input, issue one request, and validate response
shape. Chat reports `truncated: true` for `finish_reason: length`. Tool calls,
streaming, batches, image generation, and silent model fallback are unsupported.
The inference evidence means a structurally valid response to this request, not
independent proof of model identity, numerical quality, or absence of server-side
caching/fallback. Failures return a stable error code, never a raw provider body.

Input: at most 64 KiB UTF-8; chat token cap: 1–4096; HTTP response: at most 4 MiB.
A disposable stdlib worker process enforces the wall-time deadline even for slow
headers/bodies. Timeout includes worker startup (allow practical startup time).
After a deadline or cancellation the local worker is terminated; closing the HTTP
connection **cannot guarantee cancellation of server-side inference**. There is
no retry, remote cancel API, polling loop, or model load/unload request.
Python callers using multiprocessing must invoke operations under the usual
`if __name__ == '__main__':` guard in executable scripts. CLI handles this itself.

## Local speech

```sh
PYTHONPATH=src python -m agent_workbench.models whisper \
  --launcher /path/to/whisper \
  --model-path /path/to/models/large-v3-turbo \
  --audio /path/to/input.wav --output-dir /path/to/new-transcript --dry-run

PYTHONPATH=src python -m agent_workbench.models tts \
  --launcher /path/to/omnivoice_speak.sh \
  --model-path /path/to/OmniVoice-model \
  --text-file speech.txt --output-dir /path/to/new-speech --dry-run
```

Remove `--dry-run` only when ready to run inference. Dry-run validates local paths
and returns a redacted plan without starting the launcher or creating outputs.
It does not load or inspect weights, validate the audio codec, or prove installed
Python dependencies. `files: true` means required paths exist, not valid weights.
The supplied launcher is trusted executable code, not sandboxed by this adapter.

Whisper must be the installed large-v3-turbo route: explicit model directory named
`large-v3-turbo`, existing `model.bin`, `--model large-v3-turbo`,
`--local_files_only True`. CPU/int8 is default; explicit `--device cuda` selects
int8_float16 without changing model. The launcher must honor later CPU arguments
if it has its own defaults. Model identity is caller responsibility beyond these
path checks. Audio uses an absolute filename in an argument vector, not a shell.
Only a JSON artifact with a string `text` field (possibly empty for silence) is
accepted, at most 4 MiB.

OmniVoice receives `OMNIVOICE_MODEL` in the child environment, always `--no-play`,
CPU by default (`--device cuda` maps to `cuda:0`). Input text is capped at 16 KiB.
Its established CLI takes text as a positional argument, so the text is visible
in the child process argv to users permitted to inspect processes. It is not
logged by this adapter. A `--` separator prevents text becoming CLI flags.
Other launcher runtime settings, e.g. `OMNIVOICE_PYTHON`/`OMNIVOICE_LIB`, may be
supplied through the environment; no library or runtime fallback is added here.
The adapter sets Hugging Face/Transformers offline flags for both speech routes.
It validates a nonempty, nontruncated PCM WAV (at most 64 MiB), not voice quality.
No playback, microphone, camera, OGG conversion, or sending occurs.

Speech execution currently requires POSIX process groups; HTTP and dry-run plans
are portable. Launchers and their descendant group are killed on timeout,
cancellation, error, and leader exit; daemons escaping their process group are
outside this guarantee. Child stdout/stderr are discarded because they can contain
text or credentials. Diagnose a broken trusted launcher separately if needed.

The output directory must not exist and its parent must already exist. A private
staging directory contains work in progress; only the validated artifact is moved
into the reserved output directory. Failure/cancel removes staging and reservation.
Existing output is never overwritten. This contract assumes a trusted parent
directory, not adversarial concurrent filesystem mutation. SIGKILL or power loss
can leave staging/reservation directories; inspect them before removal.

## Exit codes and cancellation

Normal operations produce one JSON object on stdout; diagnostics are stable codes
on stderr. `--help` follows standard argparse human-readable output conventions.

| Exit | Meaning |
| --- | --- |
| 0 | Success (including dry-run) |
| 2 | Invalid arguments, configuration, missing local files or local I/O failure |
| 3 | Transport/provider/launcher failure or invalid output |
| 4 | Deadline exceeded |
| 130 | Cancelled via Ctrl-C, SIGTERM, or Python cancellation event |

Timeout must be finite, >0 and <=3600 seconds. Python APIs raise `ModelError`
(`code`, `status`) and accept `cancel=threading.Event()`. `main` temporarily handles
SIGTERM in the main thread and restores the caller's handler. Cancellation cleans
up local workers and child groups; it does not roll back external model servers.

## Verification boundary (2026-10-23)

Tests use a loopback mock HTTP server and tiny fixture launchers, including
malformed/partial output, credential redaction, safe argv, deadlines, cancellation,
redirect rejection, and staging cleanup. They do **not** load models.

Read-only checks on the development machine confirmed two existing local model
catalogs responded (one embedding route and one recall route). Installed Whisper
and OmniVoice paths passed dry-run checks; no ASR, TTS, embedding or chat inference
was run, no audio played, no GPU resources used. Private machine routes are
intentionally omitted here. These observations are dated evidence, not readiness
claims for another machine or a later invocation.

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_models*.py' -v
```
