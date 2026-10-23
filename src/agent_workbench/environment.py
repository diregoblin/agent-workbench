"""Read-only, dependency-free local environment observations. Import is inert."""
from __future__ import annotations

import argparse
import csv
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
from urllib.parse import urlsplit

MAX_OUTPUT = 262144
COMMANDS = {
    "python": [sys.executable, "--version"],
    "nvidia-smi": ["nvidia-smi", "--version"],
    "pw-dump": ["pw-dump", "--version"],
    "pactl": ["pactl", "--version"],
    "kscreen-doctor": ["kscreen-doctor", "--help"],
}
HINTS = {
    "missing": "Install the native tool if this capability is needed.",
    "timeout": "Check the selected service/session; do not restart it automatically.",
    "error": "Check permissions and native tool compatibility locally.",
    "malformed": "Native output did not match the supported schema.",
    "output_limit": "Native output exceeded the bounded capture limit.",
    "unavailable": "No local session socket or supported data source was found.",
    "unsupported": "This observer does not support this platform or driver.",
}


def observation(status: str, **fields) -> dict:
    result = {"status": status, **fields}
    if status in HINTS:
        result["hint"] = HINTS[status]
    return result


def run_bounded(argv: list[str], timeout: float = 3, max_output: int = MAX_OUTPUT) -> dict:
    """Internal adapter, not a shell/CLI execution interface.

    Captures bounded bytes; raw output is PRIVATE parser input, never JSON output.
    Kills only its own process group on POSIX. No inherited stdin or shell.
    """
    if not argv or not math.isfinite(timeout) or timeout <= 0 or max_output < 1:
        raise ValueError("invalid subprocess bounds")
    if not shutil.which(argv[0]):
        return observation("missing")
    env = dict(os.environ, LC_ALL="C", LANG="C")
    if Path(argv[0]).name == "kscreen-doctor":
        # Its Qt application needs a platform plugin even for --help. Avoid GUI
        # connections; the actual -j query still needs a usable KScreen session.
        env["QT_QPA_PLATFORM"] = "offscreen"
    try:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=env, shell=False,
                                 start_new_session=os.name == "posix")
    except OSError:
        return observation("error")
    chunks = [bytearray(), bytearray()]
    overflow = threading.Event()

    def drain(stream, target):
        try:
            while block := stream.read(4096):
                room = max_output - len(target)
                target.extend(block[:room])
                if len(block) > room:
                    overflow.set()
                    break
        finally:
            stream.close()

    readers = []
    status = "ok"
    try:
        for pipe, buf in zip((child.stdout, child.stderr), chunks):
            reader = threading.Thread(target=drain, args=(pipe, buf), daemon=True)
            readers.append(reader)
            reader.start()
        deadline = time.monotonic() + timeout
        while child.poll() is None or any(t.is_alive() for t in readers):
            if overflow.is_set():
                status = "output_limit"
                break
            if time.monotonic() >= deadline:
                status = "timeout"
                break
            time.sleep(0.01)
    finally:
        # Also clean up on BaseException (including cancellation and reader
        # setup failure), and when a successful leader leaves descendants with
        # closed pipes. This group belongs exclusively to this probe.
        try:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGKILL)
            elif child.poll() is None:
                child.kill()
        except ProcessLookupError:
            pass
        child.wait()
        for reader in readers:
            if reader.ident is not None:
                reader.join(timeout=0.2)
        for pipe, reader in zip((child.stdout, child.stderr), readers):
            if reader.ident is None:
                pipe.close()
        for pipe in (child.stdout, child.stderr)[len(readers):]:
            pipe.close()
    if overflow.is_set():
        status = "output_limit"
    if status == "ok" and child.returncode:
        status = "error"
    result = observation(status, returncode=child.returncode)
    # Error text can contain tokens, paths, device names or IPs: never expose it.
    if status == "ok":
        result["_stdout"] = bytes(chunks[0]).decode("utf-8", errors="replace")
    return result


def public(result: dict) -> dict:
    """Projection, not best-effort secret regex replacement."""
    return {key: value for key, value in result.items() if not key.startswith("_")}


def endpoint_probe(url: str, timeout: float) -> dict:
    """Explicit local HTTP GET only; no proxy, redirects, credentials or body output."""
    parsed = urlsplit(url)
    try:
        local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = False
    if (parsed.scheme not in {"http", "https"} or not local or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("endpoint must be a credential-free loopback HTTP(S) URL without query/fragment")
    port = parsed.port  # Validate before making any connection.
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = cls(parsed.hostname, port, timeout=timeout)
    timer = None
    started = time.monotonic()
    try:
        connection.connect()
        peer = connection.sock

        def expire():
            try:
                peer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(max(0.001, timeout - (time.monotonic() - started)), expire)
        timer.daemon = True
        timer.start()
        connection.request("GET", parsed.path or "/", headers={"Connection": "close"})
        response = connection.getresponse()
        code = response.status
        return observation("ok" if 200 <= code < 300 else "error", http_status=code,
                           reachable=True, evidence="HTTP response headers only; no inference verified")
    except (OSError, http.client.HTTPException):
        return observation("unavailable", reachable=False)
    finally:
        if timer is not None:
            timer.cancel()
        connection.close()


def doctor(timeout: float, endpoints: list[str]) -> dict:
    commands = {}
    for name, argv in COMMANDS.items():
        installed = shutil.which(argv[0]) is not None
        probe = public(run_bounded(argv, timeout))
        commands[name] = {"installed": installed, "runtime_probe": probe,
                          "evidence": "version/help command only; device/service readiness not implied"}
    return {"commands": commands,
            "endpoints": [{"index": i, **endpoint_probe(url, timeout)} for i, url in enumerate(endpoints)],
            "endpoint_scope": "only explicitly supplied endpoints; none discovered"}


def number(value: str) -> float | None:
    try:
        parsed = float(value.strip())
        return parsed if math.isfinite(parsed) and parsed >= 0 else None
    except ValueError:
        return None


def gpu_snapshot(timeout: float) -> dict:
    result = run_bounded(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], timeout)
    if result["status"] != "ok":
        return public(result)
    rows = list(csv.reader(result["_stdout"].splitlines()))
    if not rows or any(len(row) != 3 for row in rows):
        return observation("malformed")
    devices = []
    for index, row in enumerate(rows):
        if any(number(x) is None and x.strip() not in {"N/A", "[N/A]", "[Not Supported]"} for x in row):
            return observation("malformed")
        utilization, used, total = map(number, row)
        if ((utilization is not None and utilization > 100)
                or (used is not None and total is not None and used > total)):
            return observation("malformed")
        devices.append({"index": index, "utilization_percent": utilization,
                        "memory_used_mib": used, "memory_total_mib": total})
    # No process names, PIDs, users or UUIDs are queried.
    processes = run_bounded(["nvidia-smi", "--query-compute-apps=used_gpu_memory",
                             "--format=csv,noheader,nounits"], timeout)
    count = None
    if processes["status"] == "ok":
        lines = [line.strip() for line in processes["_stdout"].splitlines() if line.strip()]
        if all(number(line) is not None or line == "[N/A]" for line in lines):
            count = len(lines)
        else:
            processes = observation("malformed")
    positive = (count is not None and count > 0) or any(
        d["utilization_percent"] is not None and d["utilization_percent"] > 0 for d in devices)
    complete = count is not None and all(d["utilization_percent"] is not None for d in devices)
    busy = "busy" if positive else "no_activity_observed" if complete else "unknown"
    return observation("ok" if complete else "partial", devices=devices, busy=busy,
                       compute_process_count=count, process_probe=public(processes),
                       evidence="instantaneous NVIDIA sample, not reservation or proof of safe capacity")


def resources(timeout: float) -> dict:
    cpu = {"logical_cpu_count": os.cpu_count()}
    if hasattr(os, "getloadavg"):
        cpu["load_average"] = list(os.getloadavg())
    memory = observation("unsupported")
    if sys.platform.startswith("linux"):
        try:
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                match = re.fullmatch(r"(MemTotal|MemAvailable):\s+(\d+) kB", line)
                if match:
                    values[match[1]] = int(match[2]) * 1024
            memory = observation("ok", total_bytes=values["MemTotal"], available_bytes=values["MemAvailable"])
        except (OSError, KeyError):
            memory = observation("unavailable")
    return {"cpu": cpu, "memory": memory, "gpu": gpu_snapshot(timeout),
            "scope": "host view; container quotas and non-NVIDIA GPUs not measured"}


def json_probe(argv: list[str], timeout: float):
    result = run_bounded(argv, timeout)
    if result["status"] != "ok":
        return public(result), None
    try:
        return public(result), json.loads(result["_stdout"])
    except (ValueError, RecursionError):
        return observation("malformed"), None


def audio_inventory(timeout: float) -> dict:
    if shutil.which("pw-dump"):
        result, data = json_probe(["pw-dump", "-N", "-i", "0"], timeout)
        if result["status"] == "ok":
            if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
                return observation("malformed", backend="pipewire")
            counts = {"sinks": 0, "sources": 0}
            for item in data:
                info = item.get("info")
                props = info.get("props") if isinstance(info, dict) else None
                media_class = props.get("media.class") if isinstance(props, dict) else None
                if media_class == "Audio/Sink":
                    counts["sinks"] += 1
                elif media_class == "Audio/Source":
                    counts["sources"] += 1
            return observation("ok", backend="pipewire", **counts)
        # Keep the failure visible, do not silently switch sessions/backends.
        return {**result, "backend": "pipewire"}
    # Pin to a local existing socket, avoiding libpulse autospawn/remote defaults.
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    socket = Path(runtime) / "pulse/native" if runtime else None
    if socket is None or not socket.is_socket():
        return observation("unavailable", backend="pulseaudio")
    counts = {}
    for kind in ("sinks", "sources"):
        result, data = json_probe(["pactl", "--server", f"unix:{socket}", "--format=json", "list", kind], timeout)
        if result["status"] != "ok":
            return {**result, "backend": "pulseaudio"}
        if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
            return observation("malformed", backend="pulseaudio")
        counts[kind] = len(data)
    return observation("ok", backend="pulseaudio", **counts)


def display_inventory(timeout: float) -> dict:
    result, data = json_probe(["kscreen-doctor", "-j"], timeout)
    if result["status"] != "ok":
        return result
    if not isinstance(data, dict) or not isinstance(data.get("outputs"), list):
        return observation("malformed")
    outputs = data["outputs"]
    if any(not isinstance(item, dict) or not isinstance(item.get("connected"), bool)
           or not isinstance(item.get("enabled"), bool) for item in outputs):
        return observation("malformed")
    return observation("ok", output_count=len(outputs), connected_count=sum(x["connected"] for x in outputs),
                       enabled_count=sum(x["enabled"] for x in outputs))


def camera_inventory() -> dict:
    """Count Linux video nodes through sysfs; NEVER open a device."""
    directory = Path("/sys/class/video4linux")
    try:
        nodes = list(directory.glob("video*")) if directory.is_dir() else []
        return observation("ok", video_node_count=len(nodes),
                           evidence="sysfs nodes only; may include codecs, virtual or multiple nodes per camera")
    except OSError:
        return observation("unavailable")


def peripherals(timeout: float) -> dict:
    if not sys.platform.startswith("linux"):
        return {name: observation("unsupported") for name in ("audio", "displays", "cameras")}
    return {"audio": audio_inventory(timeout), "displays": display_inventory(timeout),
            "cameras": camera_inventory(),
            "scope": "current Linux session only; no streams opened, capture or control performed"}


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse messages can echo private input: use a constant diagnostic.
        raise ValueError("invalid arguments; use --help for supported options")


def _main(argv: list[str] | None = None) -> int:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "resources", "peripherals"])
    parser.add_argument("--timeout", type=float, default=3, help="seconds per native probe, 0.1..30")
    parser.add_argument("--endpoint", action="append", default=[], help="doctor only: explicit loopback HTTP(S) health URL")
    parser.add_argument("--config", type=Path, help="optional TOML with [environment] endpoints and timeout")
    try:
        args = parser.parse_args(argv)
        if args.config:
            with args.config.open("rb") as handle:
                config = tomllib.load(handle)
            section = config.get("environment", {})
            if not isinstance(section, dict) or set(section) - {"timeout", "endpoints"}:
                raise ValueError("unsupported environment config keys")
            # Explicit CLI timeout takes precedence over config.
            passed = list(sys.argv[1:] if argv is None else argv)
            if not any(x == "--timeout" or x.startswith("--timeout=") for x in passed):
                args.timeout = section.get("timeout", args.timeout)
            args.endpoint = section.get("endpoints", []) + args.endpoint
        if (isinstance(args.timeout, bool) or not isinstance(args.timeout, (int, float))
                or not math.isfinite(args.timeout) or not 0.1 <= args.timeout <= 30):
            raise ValueError("timeout must be between 0.1 and 30 seconds")
        if (not isinstance(args.endpoint, list) or len(args.endpoint) > 16
                or any(not isinstance(url, str) or len(url) > 2048 for url in args.endpoint)):
            raise ValueError("invalid endpoint list")
        if args.endpoint and args.command != "doctor":
            raise ValueError("endpoints are supported only by doctor")
        data = (doctor(args.timeout, args.endpoint) if args.command == "doctor" else
                resources(args.timeout) if args.command == "resources" else peripherals(args.timeout))
    except (ValueError, TypeError, OSError, AttributeError):
        print("Invalid arguments or unreadable configuration; use --help. No private input is echoed.", file=sys.stderr)
        print(json.dumps({"schema_version": 1, "status": "invalid_input"}))
        return 2

    def degraded(value):
        if isinstance(value, dict):
            return ("status" in value and value["status"] != "ok") or any(degraded(v) for v in value.values())
        return isinstance(value, list) and any(degraded(v) for v in value)

    partial = degraded(data)
    result = {"schema_version": 1, "command": args.command, "platform": platform.system(),
              "status": "partial" if partial else "ok", "observations": data}
    print(json.dumps(result, allow_nan=False))
    if partial:
        print("Some observations are unavailable or partial; inspect JSON status and hints.", file=sys.stderr)
    return 1 if partial else 0


def main(argv: list[str] | None = None) -> int:
    """CLI boundary: redact cancellation and restore the caller's signal handler."""
    def cancel(signum, frame):
        raise KeyboardInterrupt

    # Python only permits signal handlers on the main thread. Library callers
    # elsewhere retain their host's cancellation policy.
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM) if main_thread else None
    try:
        if main_thread:
            signal.signal(signal.SIGTERM, cancel)
        return _main(argv)
    except KeyboardInterrupt:
        print(json.dumps({"schema_version": 1, "status": "cancelled"}))
        print("Observation cancelled; probe cleanup completed.", file=sys.stderr)
        return 130
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
