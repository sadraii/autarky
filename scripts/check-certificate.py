#!/usr/bin/env python3
"""Check a public TLS certificate and manage one GitHub incident per endpoint.

Python 3.11+ and the standard library are sufficient for the probe. GitHub writes
also require gh and the workflow's short-lived GH_TOKEN. Use --probe-only for a
read-only check. --test-alert sends and closes a separate, labeled test issue.

Only literal IP addresses are accepted. This avoids an unbounded DNS lookup in
the TLS probe. Chain trust, certificate dates, and the IP SAN are always checked.
No private key is read. Exit codes: 0 healthy/test complete, 1 TLS problem,
2 GitHub/API failure. Argument errors also return 2.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import math
import os
import re
import socket
import ssl
import subprocess
import sys
import time
from typing import Any
from urllib.parse import quote

BOT = "github-actions[bot]"
INCIDENT_LABEL = "certificate-expiry"
TEST_LABEL = "certificate-alert-test"
PAGE_SIZE = 100
MAX_PAGES = 100
API_TIMEOUT = 15
API_BUDGET = 180
UTC = timezone.utc


class MonitorError(RuntimeError):
    """An API operation failed or could not be verified."""


@dataclass(frozen=True)
class CheckResult:
    healthy: bool
    reason: str
    checked_at: datetime
    expires_at: datetime | None = None


def inspect_certificate(cert: Any, *, checked_at: datetime, warning_hours: float) -> CheckResult:
    """Use exact UTC arithmetic. A certificate at the warning limit is a problem."""
    try:
        expiry_text = cert["notAfter"]
        if not isinstance(expiry_text, str):
            raise ValueError("Expiry must be text.")
        expires_at = datetime.fromtimestamp(ssl.cert_time_to_seconds(expiry_text), UTC)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return CheckResult(False, "The TLS certificate expiry data is not valid.", checked_at)
    remaining = expires_at - checked_at
    if remaining <= timedelta(0):
        return CheckResult(False, "The TLS certificate has expired.", checked_at, expires_at)
    if remaining <= timedelta(hours=warning_hours):
        return CheckResult(False, f"The TLS certificate expires within {warning_hours:g} hours.",
                           checked_at, expires_at)
    return CheckResult(True, f"The verified TLS certificate is valid for more than {warning_hours:g} hours.",
                       checked_at, expires_at)


def probe_certificate(host: str, port: int, warning_hours: float = 48, timeout: float = 10,
                      *, now: datetime | None = None,
                      context: ssl.SSLContext | None = None) -> CheckResult:
    # Tests can add an owned CA. They cannot turn off either trust or IP checking.
    if context is None:
        try:
            context = ssl.create_default_context()
        except (OSError, ValueError):
            return CheckResult(False, "The TLS trust store could not be loaded.", now or datetime.now(UTC))
    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        raise ValueError("TLS certificate and host verification are required.")
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            with context.wrap_socket(connection, server_hostname=host) as tls:
                cert = tls.getpeercert()
    except ssl.SSLCertVerificationError:
        reason = "TLS certificate verification failed (issuer, identity, or validity date)."
    except ssl.SSLError:
        reason = "The TLS handshake failed."
    except OSError:
        reason = "The TLS connection failed or timed out."
    except (ValueError, TypeError):
        reason = "The TLS certificate data is not valid."
    else:
        return inspect_certificate(cert, checked_at=now or datetime.now(UTC), warning_hours=warning_hours)
    # Do not include raw exception text. It can contain data from the environment.
    return CheckResult(False, reason, now or datetime.now(UTC))


class GitHub:
    """Bound each gh process and the whole API session. Never retry a write."""

    def __init__(self) -> None:
        self.deadline = time.monotonic() + API_BUDGET

    def request(self, method: str, endpoint: str, body: dict | None = None) -> Any:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise MonitorError("The GitHub API time limit was reached.")
        command = ["gh", "api", "--hostname", "github.com", "--method", method,
                   "--header", "Accept: application/vnd.github+json",
                   "--header", "X-GitHub-Api-Version: 2026-03-10", endpoint]
        if body is not None:
            command += ["--input", "-"]
        try:
            completed = subprocess.run(
                command, input=json.dumps(body) if body is not None else None,
                capture_output=True, text=True, encoding="utf-8", check=False,
                timeout=min(API_TIMEOUT, remaining),
            )
        except subprocess.TimeoutExpired:
            raise MonitorError("The GitHub API request timed out; no write was retried.") from None
        except OSError:
            raise MonitorError("The GitHub CLI could not start.") from None
        if completed.returncode != 0:
            raise MonitorError(f"The GitHub API request failed (gh exit {completed.returncode}).")
        try:
            return json.loads(completed.stdout)
        except (ValueError, TypeError):
            raise MonitorError("The GitHub API returned invalid JSON.") from None


def pages(api: GitHub, endpoint: str) -> list[dict]:
    """Read every page before any issue write. Reaching a limit is an error."""
    items = []
    separator = "&" if "?" in endpoint else "?"
    for page in range(1, MAX_PAGES + 1):
        batch = api.request("GET", f"{endpoint}{separator}per_page={PAGE_SIZE}&page={page}")
        if (not isinstance(batch, list) or len(batch) > PAGE_SIZE
                or not all(isinstance(item, dict) for item in batch)):
            raise MonitorError("The GitHub API returned an invalid page.")
        items.extend(batch)
        if len(batch) < PAGE_SIZE:
            return items
    raise MonitorError("The GitHub page limit was reached; results were not used.")


def marker(host: str, port: int, kind: str = "incident") -> str:
    return f"<!-- autarky-certificate-monitor:v1 {kind} host={host} port={port} -->"


def is_own_issue(item: Any, host: str, port: int, kind: str = "incident") -> bool:
    if not isinstance(item, dict) or "pull_request" in item:
        return False
    user = item.get("user")
    body = item.get("body")
    return (isinstance(user, dict) and user.get("login") == BOT and user.get("type") == "Bot"
            and isinstance(body, str) and marker(host, port, kind) in body.splitlines())


def issue_number(item: Any) -> int:
    number = item.get("number") if isinstance(item, dict) else None
    if type(number) is not int or number <= 0:
        raise MonitorError("The GitHub API returned an invalid issue number.")
    return number


def read_issue(api: GitHub, repo: str, number: int, host: str, port: int,
               kind: str = "incident", state: str = "open", expected: dict | None = None) -> dict:
    item = api.request("GET", f"repos/{repo}/issues/{number}")
    if (not is_own_issue(item, host, port, kind) or issue_number(item) != number
            or item.get("state") != state):
        raise MonitorError(f"Alert issue #{number} could not be verified; it was not changed again.")
    for key, value in (expected or {}).items():
        if item.get(key) != value:
            raise MonitorError(f"Alert issue #{number} did not match the requested {key}.")
    return item


def verify_target(item: dict, owner: str, label: str) -> None:
    assignees, labels = item.get("assignees"), item.get("labels")
    if (not isinstance(assignees, list)
            or not all(isinstance(entry, dict) for entry in assignees)
            or owner not in {entry.get("login") for entry in assignees}
            or not isinstance(labels, list)
            or not all(isinstance(entry, dict) for entry in labels)
            or label not in {entry.get("name") for entry in labels}
            or re.search(rf"(?<![A-Za-z0-9_-])@{re.escape(owner)}(?![A-Za-z0-9_-])", item["body"]) is None):
        raise MonitorError(f"Alert issue #{issue_number(item)} has no verified notification target or label.")


def ensure_label(api: GitHub, repo: str, name: str) -> None:
    labels = pages(api, f"repos/{repo}/labels")
    if any(item.get("name") == name for item in labels):
        return
    body = {"name": name, "color": "d93f0b" if name == INCIDENT_LABEL else "5319e7",
            "description": "TLS certificate monitor incident" if name == INCIDENT_LABEL
            else "TLS certificate monitor notification test; not a real incident"}
    api.request("POST", f"repos/{repo}/labels", body)
    readback = api.request("GET", f"repos/{repo}/labels/{quote(name, safe='')}")
    if not isinstance(readback, dict) or any(readback.get(key) != value for key, value in body.items()):
        raise MonitorError("The GitHub label write could not be verified.")


def create_alert(api: GitHub, repo: str, host: str, port: int, *, kind: str,
                 title: str, body: str) -> dict:
    owner = repo.split("/")[0]
    label = INCIDENT_LABEL if kind == "incident" else TEST_LABEL
    ensure_label(api, repo, label)
    payload = {"title": title, "body": body, "assignees": [owner], "labels": [label]}
    created = api.request("POST", f"repos/{repo}/issues", payload)
    number = issue_number(created)
    item = read_issue(api, repo, number, host, port, kind, expected={"title": title, "body": body})
    verify_target(item, owner, label)
    return item


def close_alert(api: GitHub, repo: str, item: dict, host: str, port: int,
                *, kind: str = "incident", recovery: str) -> None:
    number = issue_number(item)
    body = item["body"].rstrip() + "\n\n" + recovery + "\n"
    payload = {"state": "closed", "state_reason": "completed", "body": body}
    api.request("PATCH", f"repos/{repo}/issues/{number}", payload)
    read_issue(api, repo, number, host, port, kind, state="closed", expected=payload)


def check_details(result: CheckResult) -> str:
    details = f"Checked at (UTC): {result.checked_at.isoformat()}\n\n{result.reason}"
    if result.expires_at is not None:
        remaining = (result.expires_at - result.checked_at).total_seconds() / 3600
        details += (f"\n\nCertificate expiry (UTC): {result.expires_at.isoformat()}"
                    f"\n\nTime left: {remaining:.2f} hours.")
    return details


def synchronize_alert(api: GitHub, repo: str, host: str, port: int, result: CheckResult) -> list[int]:
    candidates = pages(api, f"repos/{repo}/issues?state=open&sort=created&direction=asc")
    # Validate issue numbers, even for unrelated records. A malformed page is not an empty result.
    for item in candidates:
        issue_number(item)
    numbers = sorted({issue_number(item) for item in candidates if is_own_issue(item, host, port)})
    owner = repo.split("/")[0]
    if not result.healthy and not numbers:
        body = (f"{marker(host, port)}\n\n@{owner}: the public TLS certificate needs a check.\n\n"
                f"Endpoint: `{host}:{port}`\n\n{check_details(result)}\n\n"
                "This issue is kept open while the problem is active. Further checks do not add comments. "
                "A verified healthy check closes this issue.\n")
        item = create_alert(api, repo, host, port, kind="incident",
                            title=f"Certificate alert: {host}:{port}", body=body)
        return [issue_number(item)]
    for number in numbers:
        # Read the exact issue again before closing it. Do not trust an old listing.
        item = read_issue(api, repo, number, host, port)
        if result.healthy:
            close_alert(api, repo, item, host, port, recovery="## Recovery\n\n" + check_details(result))
        else:
            verify_target(item, owner, INCIDENT_LABEL)
    return numbers


def send_test_alert(api: GitHub, repo: str, host: str, port: int,
                    *, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    owner = repo.split("/")[0]
    body = (f"{marker(host, port, 'test')}\n\n@{owner}: this is a TEST of GitHub notifications.\n\n"
            f"Endpoint: `{host}:{port}`\n\nTest started (UTC): {now.isoformat()}\n\n"
            "No certificate result is used. This test does not report a real TLS problem. "
            "The monitor will close this test issue after it verifies issue creation.\n")
    item = create_alert(api, repo, host, port, kind="test",
                        title=f"TEST: certificate alert for {host}:{port}", body=body)
    number = issue_number(item)
    print(f"Test alert created and verified: #{number}.", flush=True)
    close_alert(api, repo, item, host, port, kind="test",
                recovery="## Test complete\n\nThe notification test is complete. No real incident was changed.")
    print(f"Test alert closed and verified: #{number}.")
    return number


def host_argument(value: str) -> str:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Use a literal TLS server IP address, not a URL or name.") from None
    if "%" in value:
        raise argparse.ArgumentTypeError("Scoped IP addresses are not supported.")
    return value


def repo_argument(value: str) -> str:
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}", value)
            or value.split("/")[-1] in {".", ".."}):
        raise argparse.ArgumentTypeError("Use an explicit GitHub owner/repository.")
    return value


def port_argument(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("The port must be an integer from 1 to 65535.") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("The port must be from 1 to 65535.")
    return port


def hours_argument(value: str) -> float:
    try:
        hours = float(value)
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError
        timedelta(hours=hours)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError("Warning hours must be a finite, positive time interval.") from None
    return hours


def timeout_argument(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("The timeout must be greater than 0 and at most 30 seconds.") from None
    if not math.isfinite(timeout) or not 0 < timeout <= 30:
        raise argparse.ArgumentTypeError("The timeout must be greater than 0 and at most 30 seconds.")
    return timeout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=host_argument, default="40.160.103.121", help="Literal TLS server IP address.")
    parser.add_argument("--port", type=port_argument, default=443)
    parser.add_argument("--warning-hours", type=hours_argument, default=48)
    parser.add_argument("--timeout", type=timeout_argument, default=10, help="TCP/TLS timeout in seconds (at most 30).")
    parser.add_argument("--repo", type=repo_argument, default=os.environ.get("GH_REPO", "sadraii/autarky"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe-only", action="store_true", help="Check TLS without any GitHub API call.")
    mode.add_argument("--test-alert", action="store_true", help="Create, verify, and close a separate test issue; do not probe TLS.")
    args = parser.parse_args(argv)
    try:
        if args.test_alert:
            print("Test alert mode. No certificate failure is implied.")
            send_test_alert(GitHub(), args.repo, args.host, args.port)
            return 0
        result = probe_certificate(args.host, args.port, args.warning_hours, args.timeout)
        print(f"{'HEALTHY' if result.healthy else 'PROBLEM'}: {args.host}:{args.port}\n{check_details(result)}")
        if not args.probe_only:
            numbers = synchronize_alert(GitHub(), args.repo, args.host, args.port, result)
            if numbers:
                action = "Closed and verified" if result.healthy else "Open and verified"
                print(f"{action}: " + ", ".join(f"#{number}" for number in numbers))
        return 0 if result.healthy else 1
    except MonitorError as error:
        print(f"Certificate monitor failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
