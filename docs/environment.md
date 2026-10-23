# Environment and peripherals

Dependency-free Python 3.11+ adapters around existing **read-only** native tools.
No daemon, installation, scheduler, service restart, inference, stress test,
audio playback/capture, camera opening, brightness, volume or device control.
Importing the module performs no observations.

```sh
PYTHONPATH=src python -m agent_workbench.environment doctor
PYTHONPATH=src python -m agent_workbench.environment resources
PYTHONPATH=src python -m agent_workbench.environment peripherals
PYTHONPATH=src python -m agent_workbench.environment doctor --config examples/environment-local.toml
# Only after selecting an actual documented health endpoint:
PYTHONPATH=src python -m agent_workbench.environment doctor --endpoint http://127.0.0.1:8080/health
```

## Contract

`main(argv: list[str] | None = None) -> int` supports a parent dispatcher.
Ordinary calls emit one JSON object on stdout (`schema_version: 1`), with
human diagnostics on stderr. `--help` is the conventional text exception.
Exit **0** means requested observations completed, **1** means partial/missing/
unsupported/failed observations, **2** means invalid input or unreadable config.
Cancellation (Ctrl-C or SIGTERM on the main thread) returns **130** with a
redacted `cancelled` JSON object; the caller's SIGTERM handler is restored.
An installed-but-unusable command is not ready; a busy GPU is a valid successful
observation, not an exit failure. No result promises permission to use a device.

`--timeout 0.1..30` bounds **each** probe, not the whole command. Commands run
sequentially (doctor: five executable probes plus at most sixteen explicit HTTP
probes; resources: two NVIDIA queries; peripherals: one or two audio queries and
one display query). CLI timeout overrides TOML timeout. Repeated CLI endpoints
append to `[environment].endpoints`. Endpoints are accepted only by doctor.
Unrelated top-level TOML sections are ignored for shared package config; unknown
keys inside `[environment]` fail. See the placeholder example.

### `doctor`: distinguish layers of evidence

- `installed` is PATH resolution only, not a claim that the binary is trusted.
- `runtime_probe` runs fixed `--version` commands (`kscreen-doctor --help`, since
  that tool does not consistently support `--version`). This tests executable
  launch, **not** daemon, hardware, model, or inference readiness.
- `endpoints` contains indexed results of explicitly supplied loopback HTTP(S)
  GETs. No config/port discovery, proxy, authentication, redirect following,
  response body reading or response body output occurs. A 2xx proves only the
  selected URL answered; 3xx/4xx/5xx preserves `reachable: true` but is not ready.
  Paths must be known read-only health routes: HTTP GET alone does not guarantee
  a poorly designed service has no side effects. URLs are never echoed.
- No filesystem/model catalog checks are promoted into service readiness.

### `resources`: take a sample, don't allocate

Reports CPU count, available load averages, Linux total/available RAM and NVIDIA
utilization/VRAM from `nvidia-smi`. Two queries are not an atomic snapshot.
Only compute-process **count** is retained; no PID, command, owner, UUID, serial,
model name or process identity is queried. A positive utilization or compute
process count means `busy`; successful zero observations mean
`no_activity_observed`, **not** “free”. Missing/unsupported utilization or a
failed process query yields partial data and possibly `unknown`.

Memory use alone does not identify an active job. Non-NVIDIA GPU activity and
container quotas are not measured. CPU and load observations are host-level.
Nothing starts, pauses, preempts or kills an existing workload. A later job can
start immediately after the snapshot, so this is not a reservation mechanism.

### `peripherals`: enumerate without using

Linux-only, current-session observations:

| Area | Source | Public fields and limits |
| --- | --- | --- |
| Audio | `pw-dump -N -i 0`; if absent, local-socket `pactl --format=json list sinks/sources` | Counts only. PipeWire sources exclude monitor classes; PulseAudio may include monitors. Neither is a count of physical microphones. |
| Displays | `kscreen-doctor -j` | Output, connected and enabled counts only; KScreen session/backend required. Other desktop environments are not covered. |
| Cameras | `/sys/class/video4linux/video*` | Video-node count only; nodes can be virtual, codecs or multiple nodes per camera. No `/dev/video*` is opened. |

If PipeWire exists but fails, its failure remains visible instead of silently
switching backend/session. PulseAudio fallback requires an existing local runtime
socket and pins `--server` to it, avoiding remote defaults and daemon autospawn.
KScreen is launched with child-only `QT_QPA_PLATFORM=offscreen` to avoid opening a
GUI; backend errors/timeouts remain explicit and are not repaired. Unsupported
platforms return `unsupported`, not a fabricated empty inventory. No portable
control adapter is included: native typed tools already cover those operations
with device selection and authorization.

## Bounds, privacy and trust

Native subprocesses use argv lists, no shell, no stdin, C locale, per-probe
deadlines and at most 256 KiB **per stream**. Compact PipeWire output can still
exceed this on a large graph: `output_limit` is honest partial coverage. On POSIX,
a probe's **own** process group is killed in lifecycle cleanup on success,
timeout, overflow, setup failure or interruption, including lingering descendants
after the leader exits (whether or not they hold pipes). Unrelated workloads are
untouched. Descendants that deliberately escape the group are outside this
stdlib adapter's containment; SIGKILL of the CLI itself cannot run cleanup. Windows can kill
only the direct child with the stdlib adapter; peripheral enumeration is not
implemented there. Reader threads are daemonized and joins bounded.

Native success output exists only transiently as private parser input. Public
JSON is a field projection, not regex-based secret scrubbing. Stdout/stderr,
URLs, native exception messages, executable paths, arbitrary config strings,
device names and network inventory are never forwarded, even on failure. The
internal `run_bounded()` result has a private `_stdout` field on success: callers
must not serialize it; use public observers/`main`. Counts and utilization are
still operational information: sharing results is a user's decision. There is
no “raw/details” mode that could accidentally publish personal metadata.

Endpoint probes have connect/socket limits and a shutdown deadline to stop a
server dripping headers indefinitely. HTTP headers have Python's standard size
limits. Endpoints allow only literal loopback addresses or `localhost`, no
userinfo/query/fragment, with TLS verification retained. Nothing scans the LAN.
Trusted host binaries and localhost resolution remain assumptions; this tool is
not a sandbox for malicious executables. It inherits the user's environment for
native session access but never prints it. No configuration or credential is
modified. Do not substitute its JSON status for agent permission policies.

## Existing solutions and integration

The maintained upstream tools already implement discovery; this module adds a
small portable JSON projection and explicit failure semantics, not another
hardware daemon:

- [PipeWire pw-dump](https://docs.pipewire.org/page_man_pw-dump_1.html), checked
  against current upstream documentation and installed CLI help.
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html),
  wrapped using documented query flags and verified with local read-only queries.
  Online documentation retrieval was unavailable during implementation.
- [KScreen](https://invent.kde.org/plasma/kscreen) and PulseAudio CLI are existing
  platform adapters; compositor/daemon reachability is checked only on demand.
- [psutil](https://github.com/giampaolo/psutil) is a maintained alternative for
  broader cross-platform process/resource telemetry (upstream repository checked).
  It was not added because this small host/GPU snapshot needs no dependency and
  intentionally excludes process inventory.

OpenClaw integration: invoke the module with its typed `exec`/`terminal` tool,
parse stdout JSON, and treat exit 1 as **partial evidence** rather than a failed
agent action. Select the intended host explicitly when multiple nodes exist.
Use typed `nodes` for node discovery and existing authorized Home Assistant or
computer tools for user-requested device operations; this CLI neither replaces
those tools nor grants control permission. Never automatically repair/restart
services based on a failed probe. Parent package integration can call `main`.

## Verification

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_environment*.py' -v
```

Tests exercise missing commands, timeout/process-group bounds, output overflow,
nonzero exit, malformed schemas, privacy projection, inert import, HTTP status,
redirect refusal, slow HTTP headers, config/CLI precedence, unsupported platforms
and exact read-only peripheral argv. Network tests use an ephemeral loopback
fixture, never personal services. Live smoke checks use only version/help,
NVIDIA query, PipeWire enumeration, KScreen query and sysfs enumeration. No
inference or full peripheral-use validation is claimed.
