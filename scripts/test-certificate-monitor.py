#!/usr/bin/env python3
"""Test the TLS monitor. These tests do not call GitHub or public TLS services."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("certificate_monitor", ROOT / "scripts/check-certificate.py")
assert SPEC is not None and SPEC.loader is not None
monitor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = monitor
SPEC.loader.exec_module(monitor)

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)
HOST = "40.160.103.121"
PORT = 443
REPO = "sadraii/autarky"
PREFIX = f"repos/{REPO}"


def certificate(hours):
    expiry = NOW + timedelta(hours=hours)
    return {"notAfter": expiry.strftime("%b %d %H:%M:%S %Y GMT")}


def result(hours=72):
    return monitor.inspect_certificate(certificate(hours), checked_at=NOW, warning_hours=48)


def issue(number=1, *, host=HOST, port=PORT, kind="incident", author="github-actions[bot]",
          user_type="Bot", state="open", body=None):
    return {
        "number": number,
        "state": state,
        "state_reason": None,
        "title": f"Certificate alert: {host}:{port}",
        "body": body if body is not None else f"{monitor.marker(host, port, kind)}\n\n@sadraii\n",
        "user": {"login": author, "type": user_type},
        "assignees": [{"login": "sadraii"}],
        "labels": [{"name": monitor.INCIDENT_LABEL if kind == "incident" else monitor.TEST_LABEL}],
    }


class FakeGitHub:
    """A strict, in-memory REST service. It cannot run gh or access a network."""

    def __init__(self, issues=(), labels=()):
        self.issues = {item["number"]: deepcopy(item) for item in issues}
        self.labels = {name: {"name": name} for name in labels}
        self.calls = []
        self.fail: Callable[[str, str], bool] | None = None
        self.after_write: Callable[[dict], None] | None = None

    def request(self, method, endpoint, body=None):
        self.calls.append((method, endpoint, deepcopy(body)))
        if self.fail is not None and self.fail(method, endpoint):
            raise monitor.MonitorError("Injected API failure.")
        if method == "GET" and endpoint.startswith(PREFIX + "/issues?"):
            from urllib.parse import parse_qs, urlsplit
            query = parse_qs(urlsplit(endpoint).query)
            assert query["state"] == ["open"]
            assert query["sort"] == ["created"]
            assert query["direction"] == ["asc"]
            page = int(query["page"][0])
            size = int(query["per_page"][0])
            items = [item for _, item in sorted(self.issues.items()) if item["state"] == "open"]
            return deepcopy(items[(page - 1) * size:page * size])
        if method == "GET" and endpoint.startswith(PREFIX + "/labels?"):
            from urllib.parse import parse_qs, urlsplit
            query = parse_qs(urlsplit(endpoint).query)
            page = int(query["page"][0])
            size = int(query["per_page"][0])
            return deepcopy(list(self.labels.values())[(page - 1) * size:page * size])
        if endpoint == PREFIX + "/labels" and method == "POST":
            assert body is not None
            assert set(body) == {"name", "color", "description"}
            self.labels[body["name"]] = deepcopy(body)
            return deepcopy(body)
        if endpoint.startswith(PREFIX + "/labels/") and method == "GET":
            return deepcopy(self.labels[endpoint.rsplit("/", 1)[1]])
        if endpoint == PREFIX + "/issues" and method == "POST":
            assert body is not None
            assert set(body) == {"title", "body", "assignees", "labels"}
            number = max(self.issues, default=0) + 1
            created = issue(number, body=body["body"])
            created.update(title=body["title"], labels=[{"name": name} for name in body["labels"]],
                           assignees=[{"login": name} for name in body["assignees"]])
            self.issues[number] = created
            if self.after_write is not None:
                self.after_write(self.issues[number])
            return deepcopy(self.issues[number])
        if endpoint.startswith(PREFIX + "/issues/"):
            number = int(endpoint.rsplit("/", 1)[1])
            if method == "GET":
                return deepcopy(self.issues[number])
            if method == "PATCH":
                assert body is not None
                self.issues[number].update(deepcopy(body))
                if self.after_write is not None:
                    self.after_write(self.issues[number])
                return deepcopy(self.issues[number])
        raise AssertionError((method, endpoint, body))

    def writes(self, method="POST", path="/issues"):
        return [call for call in self.calls if call[0] == method and call[1].endswith(path)]


class CertificateTests(unittest.TestCase):
    def test_more_than_warning_time_is_healthy(self):
        checked = result(49)
        self.assertTrue(checked.healthy)
        self.assertEqual(checked.expires_at, NOW + timedelta(hours=49))
        self.assertEqual(checked.checked_at, NOW)
        self.assertEqual(checked.expires_at.utcoffset(), timedelta(0))

    def test_threshold_is_inclusive_and_does_not_round(self):
        for seconds, healthy in ((-1, False), (0, False), (1, True)):
            with self.subTest(seconds=seconds):
                cert = certificate(48 + seconds / 3600)
                self.assertEqual(monitor.inspect_certificate(
                    cert, checked_at=NOW, warning_hours=48).healthy, healthy)

    def test_expired_or_soon_to_expire_is_a_problem(self):
        for hours in (-1, 0, 1, 47):
            with self.subTest(hours=hours):
                self.assertFalse(result(hours).healthy)

    def test_warning_time_is_configurable(self):
        self.assertTrue(monitor.inspect_certificate(
            certificate(30), checked_at=NOW, warning_hours=24).healthy)

    def test_malformed_expiry_is_a_problem(self):
        for cert in ({}, {"notAfter": None}, {"notAfter": 42},
                     {"notAfter": "not a date"}, {"notAfter": "Jan 99 00:00:00 2026 GMT"}):
            with self.subTest(cert=cert):
                checked = monitor.inspect_certificate(cert, checked_at=NOW, warning_hours=48)
                self.assertFalse(checked.healthy)
                self.assertIsNone(checked.expires_at)

    def test_probe_requires_chain_and_ip_validation_and_uses_timeouts(self):
        context = ssl.create_default_context()
        connection = Mock()
        tls = Mock()
        tls.getpeercert.return_value = certificate(72)
        context_mock = MagicMock()
        context_mock.check_hostname = True
        context_mock.verify_mode = ssl.CERT_REQUIRED
        context_mock.wrap_socket.return_value.__enter__.return_value = tls
        with patch.object(monitor.ssl, "create_default_context", return_value=context_mock), \
                patch.object(monitor.socket, "create_connection") as connect:
            connect.return_value.__enter__.return_value = connection
            checked = monitor.probe_certificate(HOST, PORT, now=NOW)
        self.assertTrue(checked.healthy)
        connect.assert_called_once_with((HOST, PORT), timeout=10)
        context_mock.wrap_socket.assert_called_once_with(connection, server_hostname=HOST)
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_probe_does_not_accept_an_insecure_injected_context(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # This invalid context must be rejected before any network operation.
        context.check_hostname = False
        with patch.object(monitor.socket, "create_connection") as connect:
            with self.assertRaises(ValueError):
                monitor.probe_certificate(HOST, PORT, now=NOW, context=context)
        connect.assert_not_called()

    def test_tls_hostname_expiry_and_network_errors_are_problems_without_raw_error_text(self):
        errors = [ssl.SSLCertVerificationError("private token in error"),
                  ssl.SSLCertVerificationError("hostname mismatch"),
                  ssl.SSLCertVerificationError("certificate has expired"),
                  ssl.SSLError("private token in error"),
                  TimeoutError("private token in error"),
                  OSError("private token in error")]
        for error in errors:
            with self.subTest(error=type(error).__name__), \
                    patch.object(monitor.socket, "create_connection", side_effect=error):
                checked = monitor.probe_certificate(HOST, PORT, now=NOW)
                self.assertFalse(checked.healthy)
                self.assertIsNone(checked.expires_at)
                self.assertNotIn("private token", checked.reason)

    def test_a_trust_store_load_failure_is_also_a_problem(self):
        with patch.object(monitor.ssl, "create_default_context", side_effect=OSError("private-token")):
            checked = monitor.probe_certificate(HOST, PORT, now=NOW)
        self.assertFalse(checked.healthy)
        self.assertNotIn("private-token", checked.reason)


class AlertTests(unittest.TestCase):
    def test_problem_creates_assigned_mentioned_labeled_issue_and_reads_it_back(self):
        api = FakeGitHub()
        numbers = monitor.synchronize_alert(api, REPO, HOST, PORT, result(12))
        self.assertEqual(numbers, [1])
        created = api.issues[1]
        self.assertEqual(created["assignees"], [{"login": "sadraii"}])
        self.assertIn("@sadraii", created["body"])
        self.assertIn(monitor.marker(HOST, PORT), created["body"].splitlines())
        self.assertIn({"name": monitor.INCIDENT_LABEL}, created["labels"])
        self.assertEqual(api.calls[-1][:2], ("GET", PREFIX + "/issues/1"))
        self.assertEqual(len(api.writes()), 1)
        label_create = api.writes(path="/labels")
        self.assertEqual(len(label_create), 1)
        label_read = ("GET", PREFIX + "/labels/" + monitor.INCIDENT_LABEL, None)
        self.assertIn(label_read, api.calls)

    def test_an_existing_incident_causes_no_duplicate_or_comment(self):
        api = FakeGitHub([issue()])
        for _ in range(3):
            self.assertEqual(monitor.synchronize_alert(api, REPO, HOST, PORT, result(12)), [1])
        self.assertFalse([call for call in api.calls if call[0] != "GET"])

    def test_page_two_incident_prevents_duplicate_creation(self):
        unrelated = [issue(n, author="someone", user_type="User") for n in range(1, 101)]
        api = FakeGitHub(unrelated + [issue(101)])
        self.assertEqual(monitor.synchronize_alert(api, REPO, HOST, PORT, result(12)), [101])
        self.assertEqual(len([call for call in api.calls if "/issues?" in call[1]]), 2)
        self.assertFalse(api.writes())

    def test_exact_full_page_requires_a_following_page(self):
        api = FakeGitHub([issue(n) for n in range(1, 101)])
        monitor.synchronize_alert(api, REPO, HOST, PORT, result(12))
        self.assertEqual(len([call for call in api.calls if "/issues?" in call[1]]), 2)

    def test_pagination_limit_fails_instead_of_truncating(self):
        api = FakeGitHub([issue(n) for n in range(1, 101)])
        with patch.object(monitor, "MAX_PAGES", 1), self.assertRaises(monitor.MonitorError):
            monitor.synchronize_alert(api, REPO, HOST, PORT, result(12))
        self.assertFalse(api.writes())

    def test_second_page_api_failure_cannot_create_a_duplicate(self):
        api = FakeGitHub([issue(n) for n in range(1, 102)])
        api.fail = lambda method, endpoint: "page=2" in endpoint
        with self.assertRaises(monitor.MonitorError):
            monitor.synchronize_alert(api, REPO, HOST, PORT, result(12))
        self.assertFalse(api.writes())

    def test_recovery_closes_all_own_incidents_and_verifies_each_close(self):
        api = FakeGitHub([issue(1), issue(2)])
        self.assertEqual(monitor.synchronize_alert(api, REPO, HOST, PORT, result()), [1, 2])
        for number in (1, 2):
            self.assertEqual(api.issues[number]["state"], "closed")
            self.assertEqual(api.issues[number]["state_reason"], "completed")
            self.assertIn("Recovery", api.issues[number]["body"])
            self.assertIn(NOW.isoformat(), api.issues[number]["body"])
            write = next(i for i, call in enumerate(api.calls)
                         if call[:2] == ("PATCH", f"{PREFIX}/issues/{number}"))
            self.assertEqual(api.calls[write + 1][:2], ("GET", f"{PREFIX}/issues/{number}"))

    def test_unrelated_spoofed_other_endpoint_test_issues_and_prs_are_untouched(self):
        other = [issue(2, author="sadraii", user_type="User"),
                 issue(3, user_type="User"), issue(4, host="192.0.2.1"),
                 issue(5, port=8443), issue(6, kind="test"),
                 issue(7, body="A normal issue."),
                 issue(8, body="Text " + monitor.marker(HOST, PORT))]
        pull_request = issue(9)
        pull_request["pull_request"] = {"url": "fixture"}
        other.append(pull_request)
        for hours in (12, 72):
            with self.subTest(hours=hours):
                api = FakeGitHub([issue()] + other)
                monitor.synchronize_alert(api, REPO, HOST, PORT, result(hours))
                for item in other:
                    self.assertEqual(api.issues[item["number"]], item)

    def test_healthy_without_incident_does_not_write(self):
        api = FakeGitHub()
        self.assertEqual(monitor.synchronize_alert(api, REPO, HOST, PORT, result()), [])
        self.assertTrue(all(call[0] == "GET" for call in api.calls))

    def test_api_failure_is_not_an_empty_healthy_result(self):
        api = FakeGitHub()
        api.fail = lambda method, endpoint: True
        with self.assertRaises(monitor.MonitorError):
            monitor.synchronize_alert(api, REPO, HOST, PORT, result())
        self.assertFalse(api.writes())

    def test_create_failure_is_not_retried(self):
        api = FakeGitHub()
        api.fail = lambda method, endpoint: method == "POST" and endpoint.endswith("/issues")
        with self.assertRaises(monitor.MonitorError):
            monitor.synchronize_alert(api, REPO, HOST, PORT, result(12))
        self.assertEqual(len(api.writes()), 1)

    def test_readback_detects_missing_assignment_without_repeated_creation(self):
        api = FakeGitHub()
        api.after_write = lambda created: created.update(assignees=[])
        with self.assertRaises(monitor.MonitorError):
            monitor.synchronize_alert(api, REPO, HOST, PORT, result(12))
        with self.assertRaises(monitor.MonitorError):
            monitor.synchronize_alert(api, REPO, HOST, PORT, result(12))
        self.assertEqual(len(api.writes()), 1)

    def test_unverified_closure_is_an_error(self):
        api = FakeGitHub([issue()])
        api.after_write = lambda item: item.update(state="open")
        with self.assertRaises(monitor.MonitorError):
            monitor.synchronize_alert(api, REPO, HOST, PORT, result())

    def test_issue_that_changes_after_listing_is_not_closed(self):
        api = FakeGitHub([issue()])
        request = api.request

        def replace_author(method, endpoint, body=None):
            if method == "GET" and endpoint == PREFIX + "/issues/1":
                api.issues[1]["user"] = {"login": "someone", "type": "User"}
            return request(method, endpoint, body)

        api.request = replace_author
        with self.assertRaises(monitor.MonitorError):
            monitor.synchronize_alert(api, REPO, HOST, PORT, result())
        self.assertFalse([call for call in api.calls if call[0] == "PATCH"])

    def test_manual_test_uses_its_own_label_marker_and_verified_create_close(self):
        api = FakeGitHub([issue()])
        before = deepcopy(api.issues[1])
        output = io.StringIO()
        with redirect_stdout(output):
            number = monitor.send_test_alert(api, REPO, HOST, PORT, now=NOW)
        self.assertEqual(number, 2)
        self.assertEqual(api.issues[1], before)
        created = api.issues[2]
        self.assertEqual(created["state"], "closed")
        self.assertIn("TEST", created["title"])
        self.assertIn("@sadraii", created["body"])
        self.assertIn({"name": monitor.TEST_LABEL}, created["labels"])
        self.assertIn(monitor.marker(HOST, PORT, "test"), created["body"].splitlines())
        self.assertNotIn(monitor.marker(HOST, PORT), created["body"])
        self.assertIn("#2", output.getvalue())
        self.assertEqual(api.calls[-1][:2], ("GET", PREFIX + "/issues/2"))
        self.assertFalse([call for call in api.calls if "/issues?" in call[1]])


class GitHubClientTests(unittest.TestCase):
    def test_api_uses_json_stdin_no_shell_explicit_endpoint_and_bounded_timeout(self):
        payload = {"title": "A title with ' quotes", "labels": [], "assignees": ["sadraii"]}
        completed = subprocess.CompletedProcess([], 0, '{"number": 1}', "")
        with patch.object(monitor.subprocess, "run", return_value=completed) as run:
            self.assertEqual(monitor.GitHub().request("POST", PREFIX + "/issues", payload), {"number": 1})
        args, kwargs = run.call_args
        command = args[0]
        self.assertEqual(command[0:2], ["gh", "api"])
        self.assertIn("github.com", command)
        self.assertIn(PREFIX + "/issues", command)
        self.assertIn("--input", command)
        self.assertEqual(json.loads(kwargs["input"]), payload)
        self.assertFalse(kwargs.get("shell", False))
        self.assertGreater(kwargs["timeout"], 0)
        self.assertLessEqual(kwargs["timeout"], 15)

    def test_cli_failure_timeout_missing_gh_and_invalid_json_fail_with_no_secret_text(self):
        cases = [subprocess.CompletedProcess([], 1, "", "private-token"),
                 subprocess.CompletedProcess([], 0, "not JSON private-token", ""),
                 subprocess.TimeoutExpired("gh private-token", 1),
                 FileNotFoundError("private-token")]
        for case in cases:
            with self.subTest(case=type(case).__name__):
                kwargs: dict[str, Any] = ({"side_effect": case} if isinstance(case, Exception)
                                          else {"return_value": case})
                with patch.object(monitor.subprocess, "run", **kwargs):
                    with self.assertRaises(monitor.MonitorError) as error:
                        monitor.GitHub().request("GET", PREFIX + "/issues")
                self.assertNotIn("private-token", str(error.exception))

    def test_whole_api_session_has_a_deadline(self):
        with patch.object(monitor.time, "monotonic", return_value=0):
            api = monitor.GitHub()
        with patch.object(monitor.time, "monotonic", return_value=1000), \
                patch.object(monitor.subprocess, "run") as run:
            with self.assertRaises(monitor.MonitorError):
                api.request("GET", PREFIX + "/issues")
        run.assert_not_called()

    def test_invalid_page_shape_fails_instead_of_creating_an_issue(self):
        for page in ({"message": "error"}, [None], "not a page"):
            api = Mock()
            api.request.return_value = page
            with self.subTest(page=page), self.assertRaises(monitor.MonitorError):
                monitor.synchronize_alert(api, REPO, HOST, PORT, result(12))
            self.assertTrue(all(call.args[0] == "GET" for call in api.request.call_args_list))


class CommandTests(unittest.TestCase):
    def call_main(self, args, checked=None, api=None):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error), \
                patch.object(monitor, "probe_certificate", return_value=checked or result()) as probe, \
                patch.object(monitor, "GitHub", return_value=api or FakeGitHub()) as github:
            code = monitor.main(args)
        return code, output.getvalue(), error.getvalue(), probe, github

    def test_probe_only_has_no_github_calls_and_returns_real_health_status(self):
        for hours, expected in ((72, 0), (12, 1)):
            with self.subTest(hours=hours):
                code, output, error, probe, github = self.call_main(["--probe-only"], result(hours))
                self.assertEqual(code, expected, error)
                probe.assert_called_once()
                github.assert_not_called()
                self.assertIn(HOST, output)

    def test_an_active_alert_fails_the_workflow_and_healthy_recovery_succeeds(self):
        api = FakeGitHub()
        code, _, error, _, _ = self.call_main(["--repo", REPO], result(12), api)
        self.assertEqual(code, 1, error)
        code, _, error, _, _ = self.call_main(["--repo", REPO], result(), api)
        self.assertEqual(code, 0, error)
        self.assertEqual(api.issues[1]["state"], "closed")

    def test_github_api_failure_is_a_nonzero_command_result(self):
        api = FakeGitHub()
        api.fail = lambda method, endpoint: True
        code, _, error, _, _ = self.call_main(["--repo", REPO], result(), api)
        self.assertEqual(code, 2)
        self.assertIn("failed", error.lower())
        self.assertFalse(api.writes())

    def test_manual_test_does_not_probe_or_claim_a_real_failure(self):
        code, output, error, probe, _ = self.call_main(["--test-alert", "--repo", REPO])
        self.assertEqual(code, 0, error)
        probe.assert_not_called()
        self.assertIn("Test alert", output)
        self.assertIn("#1", output)
        self.assertIn("closed", output)

    def test_bad_arguments_and_insecure_mode_are_rejected_before_network_calls(self):
        cases = [["--warning-hours", "nan"], ["--warning-hours", "inf"],
                 ["--warning-hours", "0"], ["--warning-hours", "-1"],
                 ["--port", "0"], ["--port", "65536"], ["--timeout", "0"],
                 ["--timeout", "31"], ["--host", "https://private-token@example.test"],
                 ["--repo", "../other"], ["--insecure"], ["--probe-only", "--test-alert"]]
        for args in cases:
            with self.subTest(args=args), redirect_stderr(io.StringIO()), \
                    patch.object(monitor, "probe_certificate") as probe, \
                    patch.object(monitor, "GitHub") as github, self.assertRaises(SystemExit):
                monitor.main(args)
            probe.assert_not_called()
            github.assert_not_called()


class WorkflowTests(unittest.TestCase):
    def test_workflow_has_limited_permissions_safe_manual_input_and_serial_runs(self):
        workflow = (ROOT / ".github/workflows/certificate-expiry.yml").read_text(encoding="utf-8")
        for required in ("name: Certificate expiry", "cron: '17 */6 * * *'", "workflow_dispatch:",
                         "type: boolean", "default: false", "ubuntu-24.04", "timeout-minutes: 5",
                         "contents: read", "issues: write", "group: certificate-expiry-monitor",
                         "cancel-in-progress: false", "persist-credentials: false",
                         "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
                         "GH_TOKEN: ${{ github.token }}", "GH_REPO: ${{ github.repository }}",
                         '"$TEST_ALERT"', "--test-alert", "--warning-hours 48", "--port 443",
                         "60 days", "delayed", "refs/heads/master"):
            self.assertIn(required, workflow)
        run = workflow.split("        run: |", 1)[1]
        self.assertNotIn("${{", run, "Do not put GitHub expressions in shell code.")
        self.assertNotIn("secrets.", workflow)
        self.assertNotIn("pull_request", workflow)

    def test_both_workflow_shell_paths_pass_literal_arguments_and_propagate_failure(self):
        workflow = (ROOT / ".github/workflows/certificate-expiry.yml").read_text(encoding="utf-8")
        run = workflow.split("        run: |\n", 1)[1]
        script = "\n".join(line.removeprefix("          ") for line in run.splitlines())
        for test_alert in ("true", "false"):
            with self.subTest(test_alert=test_alert), tempfile.TemporaryDirectory(
                    prefix="autarky-workflow-test-") as temporary:
                root = Path(temporary)
                recorder = root / "python3"
                recorder.write_text(
                    "#!" + sys.executable + "\nimport json, os, pathlib, sys\n"
                    "pathlib.Path(os.environ['CALL_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
                    "sys.exit(7)\n", encoding="utf-8")
                recorder.chmod(0o755)
                # The shell must pass this as data. The stub does not run the monitor.
                repo = "sadraii/autarky; $(exit 99)"
                env = dict(os.environ, PATH=str(root), TEST_ALERT=test_alert, GH_REPO=repo,
                           CALL_LOG=str(root / "calls"))
                completed = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-c", script],
                                           env=env, capture_output=True, text=True, timeout=5)
                self.assertEqual(completed.returncode, 7, completed.stderr)
                args = json.loads((root / "calls").read_text(encoding="utf-8"))
                expected = ["scripts/check-certificate.py", "--host", HOST, "--port", "443",
                            "--warning-hours", "48", "--repo", repo]
                if test_alert == "true":
                    expected.append("--test-alert")
                self.assertEqual(args, expected)


@contextmanager
def local_tls_server(context):
    """Own one loopback connection and propagate unexpected server failures."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    failures = []

    def serve():
        try:
            connection, _ = listener.accept()
            with connection:
                try:
                    connection.settimeout(3)
                    with context.wrap_socket(connection, server_side=True):
                        pass  # A certificate probe only needs the TLS handshake.
                except ssl.SSLError:
                    pass  # Client rejection is required in the negative TLS tests.
        except Exception as error:
            failures.append(error)
        finally:
            listener.close()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield listener.getsockname()[1]
    finally:
        worker.join(6)
        if worker.is_alive():
            raise AssertionError("The local TLS server did not stop.")
        if failures:
            raise failures[0]


class LocalTLSIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="autarky-certificate-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        key, cert = root / "key.pem", root / "cert.pem"
        command = ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3",
                   "-subj", "/CN=autarky-owned-test", "-addext", "subjectAltName=IP:127.0.0.1",
                   "-keyout", str(key), "-out", str(cert)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
        cls.server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        cls.server.load_cert_chain(str(cert), str(key))
        cls.trusted = ssl.create_default_context(cafile=str(cert))
        cls.cert = cert

        csr, expired = root / "expired.csr", root / "expired.pem"
        subprocess.run(["openssl", "req", "-new", "-key", str(key), "-out", str(csr),
                        "-subj", "/CN=autarky-owned-test", "-addext", "subjectAltName=IP:127.0.0.1"],
                       check=True, capture_output=True, text=True, timeout=15)
        subprocess.run(["openssl", "x509", "-req", "-in", str(csr), "-signkey", str(key),
                        "-days", "-1", "-copy_extensions", "copy", "-out", str(expired)],
                       check=True, capture_output=True, text=True, timeout=15)
        cls.expired_server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        cls.expired_server.load_cert_chain(str(expired), str(key))
        cls.expired_trust = ssl.create_default_context(cafile=str(expired))

    def test_real_trusted_ip_tls_handshake_reads_certificate_expiry(self):
        with local_tls_server(self.server) as port:
            checked = monitor.probe_certificate("127.0.0.1", port, context=self.trusted)
        self.assertTrue(checked.healthy, checked.reason)
        self.assertIsNotNone(checked.expires_at)
        self.assertTrue(self.trusted.check_hostname)
        self.assertEqual(self.trusted.verify_mode, ssl.CERT_REQUIRED)

    def test_real_handshake_can_trigger_warning_without_disabling_trust(self):
        with local_tls_server(self.server) as port:
            checked = monitor.probe_certificate("127.0.0.1", port, warning_hours=96, context=self.trusted)
        self.assertFalse(checked.healthy)
        self.assertIsNotNone(checked.expires_at)

    def test_real_untrusted_issuer_is_rejected(self):
        with local_tls_server(self.server) as port:
            checked = monitor.probe_certificate("127.0.0.1", port)
        self.assertFalse(checked.healthy)
        self.assertIsNone(checked.expires_at)

    def test_real_wrong_ip_san_is_rejected(self):
        # Keep the real loopback connection. Only the requested TLS identity changes.
        real_connect = socket.create_connection
        with local_tls_server(self.server) as port, \
                patch.object(monitor.socket, "create_connection", side_effect=lambda address, timeout:
                             real_connect(("127.0.0.1", port), timeout=timeout)):
            checked = monitor.probe_certificate("127.0.0.2", port, context=self.trusted)
        self.assertFalse(checked.healthy)
        self.assertIsNone(checked.expires_at)

    def test_real_expired_certificate_is_rejected(self):
        with local_tls_server(self.expired_server) as port:
            checked = monitor.probe_certificate("127.0.0.1", port, context=self.expired_trust)
        self.assertFalse(checked.healthy)
        self.assertIsNone(checked.expires_at)

    def test_real_probe_only_cli_uses_system_verification_and_cannot_call_gh(self):
        for warning, code, status in ((48, 0, "HEALTHY"), (96, 1, "PROBLEM")):
            with self.subTest(warning=warning), local_tls_server(self.server) as port:
                # The owned trust certificate is the only test override. Empty PATH
                # also prevents this child process from running any gh command.
                env = dict(os.environ, SSL_CERT_FILE=str(self.cert), PATH="")
                command = [sys.executable, "-B", str(ROOT / "scripts/check-certificate.py"),
                           "--probe-only", "--host", "127.0.0.1", "--port", str(port),
                           "--warning-hours", str(warning)]
                completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=10)
            self.assertEqual(completed.returncode, code, completed.stderr)
            self.assertIn(status + ": 127.0.0.1:", completed.stdout)
            self.assertIn("Certificate expiry (UTC)", completed.stdout)


if __name__ == "__main__":
    unittest.main()
