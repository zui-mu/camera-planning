"""Explicit, resumable execution of an inspected YSynthetic command plan."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import write_json


def execute_command_plan(path, acknowledge_heavy=False, resume=False):
    if not acknowledge_heavy:
        raise ValueError("pass --acknowledge-heavy after inspecting the command plan")
    source = Path(path).resolve()
    plan = json.loads(source.read_text(encoding="utf-8"))
    if plan.get("execution") != "NOT_EXECUTED":
        raise ValueError("unsupported or already-mutated command plan")
    root = Path(plan["cwd"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    log_path = root / "integration_execution.json"
    previous = {}
    if log_path.exists():
        if not resume:
            raise FileExistsError(
                "execution log exists; pass --resume to continue successful stages"
            )
        previous = {
            item["stage"]: item
            for item in json.loads(log_path.read_text(encoding="utf-8")).get("stages", [])
        }
    environment = os.environ.copy()
    environment.update(plan.get("environment", {}))
    records = []
    report = {
        "schema_version": "camera_integration_execution_v1",
        "command_plan": str(source),
        "status": "running",
        "stages": records,
    }
    for command in plan["commands"]:
        stage = command["stage"]
        if previous.get(stage, {}).get("returncode") == 0:
            records.append(previous[stage])
            continue
        started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(
            command["argv"],
            cwd=root,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        stdout_path = root / "logs" / f"{stage.replace(':', '__')}.stdout.log"
        stderr_path = root / "logs" / f"{stage.replace(':', '__')}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        record = {
            "stage": stage,
            "argv": command["argv"],
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "returncode": completed.returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        records.append(record)
        report["status"] = "failed" if completed.returncode else "running"
        write_json(log_path, report)
        if completed.returncode:
            raise RuntimeError(f"stage failed: {stage}; inspect {stderr_path}")
    report["status"] = "complete"
    write_json(log_path, report)
    return report
