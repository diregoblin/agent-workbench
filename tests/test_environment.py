import contextlib
import http.server
import io
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent_workbench import environment as env


class ProcessTests(unittest.TestCase):
    def test_missing(self):
        with patch.object(env.shutil, 'which', return_value=None):
            self.assertEqual(env.run_bounded(['missing'])['status'], 'missing')

    def test_error_does_not_echo_secret(self):
        result = env.run_bounded([sys.executable, '-c', 'import sys; print("token=TOPSECRET"); sys.exit(7)'])
        self.assertEqual(result['returncode'], 7)
        self.assertEqual(result['status'], 'error')
        self.assertNotIn('TOPSECRET', json.dumps(result))

    def test_success_projection_is_redacted(self):
        result = env.run_bounded([sys.executable, '-c', 'print("Bearer TOPSECRET /private/path 192.168.1.42")'])
        self.assertIn('TOPSECRET', result['_stdout'])
        self.assertNotIn('TOPSECRET', json.dumps(env.public(result)))
        self.assertNotIn('192.168', json.dumps(env.public(result)))

    def test_timeout(self):
        start = time.monotonic()
        result = env.run_bounded([sys.executable, '-c', 'import time; time.sleep(10)'], .1)
        self.assertEqual(result['status'], 'timeout')
        self.assertLess(time.monotonic() - start, 2)

    def test_output_limit(self):
        result = env.run_bounded([sys.executable, '-c', 'print("x" * 100000)'], 2, 4096)
        self.assertEqual(result['status'], 'output_limit')
        self.assertNotIn('_stdout', result)

    @unittest.skipUnless(os.name == 'posix', 'POSIX process group behavior')
    def test_descendant_holding_pipe_is_bounded(self):
        code = 'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(10)"])'
        start = time.monotonic()
        result = env.run_bounded([sys.executable, '-c', code], .15)
        self.assertEqual(result['status'], 'timeout')
        self.assertLess(time.monotonic() - start, 2)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process state check')
    def test_real_cli_interruption_cleans_group(self):
        # These are disposable Python fixtures, never host services.
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory() as directory:
                marker = Path(directory) / 'ready.json'
                child_code = (
                    'import subprocess,sys,time,os,json; from pathlib import Path; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                    f'Path({str(marker)!r}).write_text(json.dumps([os.getpid(),p.pid])); '
                    'print("PRIVATE TOKEN",flush=True); time.sleep(30)'
                )
                driver = (
                    'import sys; from agent_workbench import environment as e; '
                    f'e.doctor=lambda *a: e.run_bounded([sys.executable,"-c",{child_code!r}],30); '
                    'sys.exit(e.main(["doctor"]))'
                )
                process = subprocess.Popen([sys.executable, '-c', driver],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                pids = []
                try:
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        try:
                            pids = json.loads(marker.read_text())
                            break
                        except (FileNotFoundError, json.JSONDecodeError):
                            time.sleep(.01)
                    self.assertEqual(len(pids), 2, 'fixture did not start')
                    process.send_signal(sig)
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 130)
                    self.assertEqual(json.loads(stdout), {'schema_version': 1, 'status': 'cancelled'})
                    self.assertNotIn(b'PRIVATE', stdout + stderr)
                    self.assertNotIn(b'Traceback', stderr)
                    for pid in pids:
                        self.assert_process_stopped(pid)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)
                    for pid in pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def assert_process_stopped(self, pid):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                # An orphan zombie awaits the host reaper; it cannot execute.
                state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
                if state == 'Z':
                    return
            except FileNotFoundError:
                return
            time.sleep(.01)
        self.fail('probe process remained running')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process state check')
    def test_successful_leader_cleans_detached_pipe_descendant(self):
        code = (
            'import subprocess,sys; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],'
            'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(p.pid)'
        )
        result = env.run_bounded([sys.executable, '-c', code], 2)
        self.assertEqual(result['status'], 'ok')
        pid = int(result['_stdout'])
        try:
            self.assert_process_stopped(pid)
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_reader_setup_failure_reaps_child(self):
        original = env.subprocess.Popen
        children = []
        def capture(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child
        with patch.object(env.subprocess, 'Popen', side_effect=capture), patch.object(
                env.threading.Thread, 'start', side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):
                env.run_bounded([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.assertIsNotNone(children[0].returncode)
        self.assertTrue(children[0].stdout.closed)
        self.assertTrue(children[0].stderr.closed)

    def test_import_is_inert(self):
        code = '''import subprocess, socket, ssl
subprocess.Popen = lambda *a, **k: (_ for _ in ()).throw(AssertionError('spawn'))
socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network'))
import agent_workbench.environment
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b'')


class ObservationTests(unittest.TestCase):
    def test_gpu_busy_and_no_identity(self):
        responses = [{'status': 'ok', '_stdout': '90, 6000, 8000\n'},
                     {'status': 'ok', '_stdout': '5500\n'}]
        with patch.object(env, 'run_bounded', side_effect=responses) as run:
            result = env.gpu_snapshot(1)
        self.assertEqual(result['busy'], 'busy')
        self.assertEqual(result['compute_process_count'], 1)
        self.assertEqual(run.call_args_list[1].args[0][1], '--query-compute-apps=used_gpu_memory')
        self.assertNotIn('_stdout', json.dumps(result))

    def test_gpu_no_activity_is_not_free_claim(self):
        with patch.object(env, 'run_bounded', side_effect=[
            {'status': 'ok', '_stdout': '0, 800, 8000'}, {'status': 'ok', '_stdout': ''}]):
            result = env.gpu_snapshot(1)
        self.assertEqual(result['busy'], 'no_activity_observed')
        self.assertIn('not reservation', result['evidence'])

    def test_gpu_unknown(self):
        with patch.object(env, 'run_bounded', side_effect=[
            {'status': 'ok', '_stdout': '[N/A], 800, 8000'}, {'status': 'error'}]):
            result = env.gpu_snapshot(1)
        self.assertEqual(result['busy'], 'unknown')
        self.assertEqual(result['status'], 'partial')

    def test_gpu_malformed(self):
        for output in ['garbage', 'garbage, 1, 2', '101, 1, 2', '0, 9, 2', '']:
            with self.subTest(output=output), patch.object(env, 'run_bounded', return_value={'status': 'ok', '_stdout': output}):
                self.assertEqual(env.gpu_snapshot(1)['status'], 'malformed')

    def test_json_malformed(self):
        with patch.object(env, 'run_bounded', return_value={'status': 'ok', '_stdout': 'not-json TOPSECRET'}):
            self.assertEqual(env.json_probe(['unused'], 1), ({'status': 'malformed', 'hint': env.HINTS['malformed']}, None))

    def test_audio_projects_counts_only(self):
        data = [{'info': {'props': {'media.class': 'Audio/Sink', 'node.name': 'PRIVATE'}}},
                {'info': {'props': {'media.class': 'Audio/Source'}}}, {'info': None}]
        with patch.object(env.shutil, 'which', return_value='/tool'), patch.object(env, 'json_probe', return_value=({'status': 'ok'}, data)) as probe:
            result = env.audio_inventory(1)
        self.assertEqual(result['sinks'], 1)
        self.assertEqual(result['sources'], 1)
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertEqual(probe.call_args.args[0], ['pw-dump', '-N', '-i', '0'])

    def test_audio_malformed(self):
        with patch.object(env.shutil, 'which', return_value='/tool'), patch.object(env, 'json_probe', return_value=({'status': 'ok'}, ['bad'])):
            self.assertEqual(env.audio_inventory(1)['status'], 'malformed')

    def test_pulse_pins_socket_and_only_lists(self):
        with patch.object(env.shutil, 'which', return_value=None), patch.dict(os.environ, {'XDG_RUNTIME_DIR': '/runtime'}), patch.object(Path, 'is_socket', return_value=True), patch.object(env, 'json_probe', return_value=({'status': 'ok'}, [{}])) as probe:
            result = env.audio_inventory(1)
        self.assertEqual(result['sources'], 1)
        self.assertEqual([call.args[0] for call in probe.call_args_list], [
            ['pactl', '--server', 'unix:/runtime/pulse/native', '--format=json', 'list', kind]
            for kind in ('sinks', 'sources')])

    def test_display_only_queries(self):
        with patch.object(env, 'json_probe', return_value=({'status': 'ok'}, {'outputs': [{'connected': True, 'enabled': False, 'name': 'PRIVATE'}]})) as probe:
            result = env.display_inventory(1)
        self.assertEqual(probe.call_args.args[0], ['kscreen-doctor', '-j'])
        self.assertEqual(result['connected_count'], 1)
        self.assertNotIn('PRIVATE', json.dumps(result))

    def test_display_malformed(self):
        with patch.object(env, 'json_probe', return_value=({'status': 'ok'}, {'outputs': [{}]})):
            self.assertEqual(env.display_inventory(1)['status'], 'malformed')

    def test_camera_never_opens_device(self):
        with patch.object(Path, 'is_dir', return_value=True), patch.object(Path, 'glob', return_value=[Path('/sys/class/video4linux/video0')]), patch('builtins.open', side_effect=AssertionError('opened')):
            self.assertEqual(env.camera_inventory()['video_node_count'], 1)

    def test_unsupported_platform(self):
        with patch.object(env.sys, 'platform', 'win32'), patch.object(env, 'run_bounded', side_effect=AssertionError('spawn')):
            self.assertTrue(all(x['status'] == 'unsupported' for x in env.peripherals(1).values()))


class EndpointTests(unittest.TestCase):
    def test_reject_remote_credentials_queries(self):
        for url in ['http://example.com/', 'http://127.0.0.1/?token=secret', 'http://user:pass@localhost/', 'file:///tmp/x', 'http://127.0.0.1/#secret']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                env.endpoint_probe(url, .1)

    def test_http_status_and_no_redirect(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302 if self.path == '/redirect' else 200)
                self.send_header('Location', 'http://example.com/PRIVATE')
                self.end_headers()
                self.wfile.write(b'PRIVATE TOKEN')
            def log_message(self, *args):
                pass
        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}'
            self.assertEqual(env.endpoint_probe(base, 1)['status'], 'ok')
            result = env.endpoint_probe(base + '/redirect', 1)
            self.assertEqual(result['http_status'], 302)
            self.assertEqual(result['status'], 'error')
            self.assertNotIn('PRIVATE', json.dumps(result))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_slow_headers_total_deadline(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    for _ in range(30):
                        self.wfile.write(b'x')
                        self.wfile.flush()
                        time.sleep(.02)
                except OSError:
                    pass
            def log_message(self, *args):
                pass
        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            started = time.monotonic()
            result = env.endpoint_probe(f'http://127.0.0.1:{server.server_port}', .1)
            self.assertEqual(result['status'], 'unavailable')
            self.assertLess(time.monotonic() - started, .5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class CliTests(unittest.TestCase):
    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = env.main(args)
        return code, json.loads(stdout.getvalue()), stderr.getvalue()

    def test_cancel_restores_signal_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        with patch.object(env, 'doctor', side_effect=KeyboardInterrupt('PRIVATE')):
            code, result, stderr = self.invoke(['doctor'])
        self.assertEqual(code, 130)
        self.assertEqual(result['status'], 'cancelled')
        self.assertNotIn('PRIVATE', stderr)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_invalid_input_redacted(self):
        for args in [['PRIVATE'], ['doctor', '--timeout', 'nan'], ['doctor', '--config', '/PRIVATE/TOKEN']]:
            code, result, stderr = self.invoke(args)
            self.assertEqual(code, 2)
            self.assertEqual(result['status'], 'invalid_input')
            self.assertNotIn('PRIVATE', stderr + json.dumps(result))

    def test_doctor_separates_installed_runtime(self):
        with patch.object(env.shutil, 'which', return_value='/PRIVATE'), patch.object(env, 'run_bounded', return_value={'status': 'error'}):
            code, result, _ = self.invoke(['doctor'])
        command = result['observations']['commands']['python']
        self.assertTrue(command['installed'])
        self.assertEqual(command['runtime_probe']['status'], 'error')
        self.assertEqual(code, 1)
        self.assertNotIn('PRIVATE', json.dumps(result))

    def test_config_and_cli_timeout_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'example.toml'
            config.write_text('[environment]\ntimeout = 2\nendpoints = []\n')
            with patch.object(env, 'doctor', return_value={}) as doctor:
                self.assertEqual(self.invoke(['doctor', '--config', str(config), '--timeout', '1'])[0], 0)
                doctor.assert_called_once_with(1., [])

    def test_endpoint_limit(self):
        code, _, _ = self.invoke(['doctor'] + ['--endpoint', 'http://localhost/'] * 17)
        self.assertEqual(code, 2)


if __name__ == '__main__':
    unittest.main()
