# -*- coding: utf-8 -*-
"""
Keep scheduled GitHub Actions workflows alive over the off-season.

GitHub disables scheduled workflows in public repos after 60 days without
repository activity. Between the last race (early December) and the first
race (March) nothing gets committed, so both the race-weekend workflow here
and update-f1.yml on the website would be switched off.

For each repo, if the newest commit on the default branch is older than
STALE_DAYS, this writes a timestamp to .github/keepalive.txt through the
GitHub contents API (one commit, no local clone touched). Uses the
authenticated `gh` CLI.

Run weekly from Windows Task Scheduler (task "F1 keepalive"):
    python scripts/keepalive.py
    python scripts/keepalive.py --dry-run
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOS = ["leo-01000111/F1-Predictor", "leo-01000111/leo-01000111.github.io"]
STALE_DAYS = 45  # well inside the 60-day limit, so a few missed weeks are fine
PATH = ".github/keepalive.txt"
MESSAGE = "chore: keep scheduled workflows alive [skip ci]"
LOG = Path(__file__).resolve().parent.parent / "logs" / "keepalive.log"


def gh_api(*args: str) -> dict | list:
    out = subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout) if out.stdout.strip() else {}


def last_commit_age_days(repo: str) -> float:
    commit = gh_api(f"repos/{repo}/commits?per_page=1")[0]
    when = datetime.fromisoformat(commit["commit"]["committer"]["date"].replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400


def touch(repo: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = base64.b64encode(f"{now}\n".encode()).decode()
    args = [f"repos/{repo}/contents/{PATH}", "-X", "PUT",
            "-f", f"message={MESSAGE}", "-f", f"content={content}"]
    try:
        sha = gh_api(f"repos/{repo}/contents/{PATH}")["sha"]
        args += ["-f", f"sha={sha}"]
    except subprocess.CalledProcessError:
        pass  # file does not exist yet
    gh_api(*args)


def log(line: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(line)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{stamp} {line}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="report only, commit nothing")
    args = parser.parse_args()

    failed = False
    for repo in REPOS:
        try:
            age = last_commit_age_days(repo)
            if age < STALE_DAYS:
                log(f"{repo}: last commit {age:.0f} days ago, nothing to do")
            elif args.dry_run:
                log(f"{repo}: last commit {age:.0f} days ago, would commit (dry run)")
            else:
                touch(repo)
                log(f"{repo}: last commit {age:.0f} days ago, keep-alive commit pushed")
        except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as exc:
            failed = True
            detail = getattr(exc, "stderr", "") or exc
            log(f"{repo}: FAILED {str(detail).strip()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
