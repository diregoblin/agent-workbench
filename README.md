# Agent Workbench

[![Check](https://github.com/diregoblin/agent-workbench/actions/workflows/check.yml/badge.svg)](https://github.com/diregoblin/agent-workbench/actions/workflows/check.yml)

Small, inspectable CLI adapters for an agent's local environment, peripherals,
and auxiliary models. **Python 3.11+, no runtime dependencies, MIT.** Existing
native tools and model servers do the work; this is not another agent runtime.

## Install

```sh
git clone https://github.com/diregoblin/agent-workbench.git
cd agent-workbench
python3 -m venv .venv
.venv/bin/python -m pip install --no-deps .
.venv/bin/agent-workbench --help
```

Alternatively use `pipx install .` or run directly from the checkout with
`PYTHONPATH=src python3 -m agent_workbench`. On Windows, the virtual environment's
executables live under `.venv/Scripts/`.

## A small toolbox

| Command | What it does | What it does not prove |
| --- | --- | --- |
| `environment doctor` | Check native executables and optional loopback health URLs | Device or model readiness |
| `environment resources` | Read CPU, RAM and NVIDIA activity | Exclusive GPU availability |
| `environment peripherals` | Count Linux audio nodes, KScreen outputs and video nodes | Capture, playback or physical device identity |
| `models health` / `models list` | Query an explicit local OpenAI-compatible catalog | Successful inference |
| `models chat` / `models embed` | Make one bounded request to an explicit model | Model quality or server-side cancellation |
| `models whisper` | Invoke an installed large-v3-turbo launcher | Microphone quality |
| `models tts` | Invoke an installed OmniVoice-compatible launcher without playback | Voice quality |

```sh
agent-workbench environment doctor
agent-workbench environment resources
agent-workbench environment peripherals --timeout 2
agent-workbench models list --endpoint http://127.0.0.1:8080/v1

# Explicit inference; the server must already be running:
agent-workbench models chat --endpoint http://127.0.0.1:8080/v1 \
  --model YOUR_MODEL_ID --text-file prompt.txt --max-tokens 128 --timeout 30

# Validate local paths without loading a speech model:
agent-workbench models whisper --launcher /path/to/whisper \
  --model-path /path/to/models/large-v3-turbo --audio /path/to/input.wav \
  --output-dir /path/to/new-transcript --dry-run
```

Normal operations print JSON to stdout and short diagnostics to stderr. Imports
and `--help` are inert. Environment commands use exits **0 / 1 / 2** for complete /
partial / invalid observations; model commands additionally distinguish provider
failure, deadline and cancellation. Do not interpret partial device coverage as
an empty inventory. See the exact contracts in
[environment and peripherals](docs/environment.md) and [models](docs/models.md).

## Deliberately bounded

- No services, schedulers, model downloads, model switching or installation.
- No device activation, recording, playback, volume or display control.
- No shell interpolation; explicit deadlines and bounded output.
- HTTP uses loopback only, no proxies, redirects or retries. Optional API keys
  come from a named environment variable, not a command-line credential.
- Speech defaults to CPU, existing offline models and new artifact directories.
- Diagnostics project counts and status rather than raw device names or logs.
  Inference output and local artifact paths can still contain private data.
- Never stops another workload or treats a resource snapshot as authorization.

The HTTP adapter and CLI are portable. Linux provides the richest environment
coverage; speech execution currently requires POSIX process groups. Native tools
remain optional: missing tools are reported rather than installed.

## Integrate with your agent

Call the CLI through your existing process tool and parse JSON. The dispatcher
also exposes `main(argv)`; modules offer direct Python functions. No MCP server,
OpenClaw plugin, background daemon or global configuration is required.

[Example TOML files](examples/) contain placeholders only. The environment module
can load its `[environment]` section; model examples are an argument-mapping
reference, **not** automatically loaded configuration.

## Verify

```sh
python -m pip install --no-deps .
python -m unittest discover -s tests -v
agent-workbench environment --help
agent-workbench models --help
```

Tests use loopback HTTP fixtures and tiny local executables, not real inference
or device control. Live development checks established local catalogs, NVIDIA
activity and audio/video-node enumeration. KScreen timed out and remains partial;
speech paths were dry-run checked only. See module documentation for boundaries.
