"""Collect completed YSynthetic body-fusion results into auditable RQ1 tables."""

import csv
import json
from pathlib import Path

from .artifacts import file_hash, write_json
from .evaluation import evaluate_fused


def _find_body_fuse_result(set_dir):
    matches = sorted((set_dir / "outputs").glob("**/body_fuse_result.json"))
    if len(matches) > 1:
        raise ValueError(f"multiple body_fuse_result.json files under {set_dir}")
    return matches[0] if matches else None


def summarize_experiment(experiment_dir, output=None):
    root = Path(experiment_dir).resolve()
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    reference = Path(manifest["reference"]["path"])
    if file_hash(reference) != manifest["reference"]["sha256"]:
        raise ValueError("reference artifact changed since experiment preparation")
    rows, canonical = [], {}
    for entry in manifest["sets"]:
        row = {
            "set_id": entry["set_id"],
            "method": entry["method"],
            "seed": entry["seed"],
            "duplicate_of": entry.get("duplicate_of"),
            "camera_ids": entry["camera_ids"],
            **{f"proxy_{key}": value for key, value in entry["proxy_metrics"].items()},
        }
        source_id = entry.get("duplicate_of") or entry["set_id"]
        if source_id in canonical:
            row.update(canonical[source_id])
            row["status"] = "duplicate_reused"
            rows.append(row)
            continue
        result_path = _find_body_fuse_result(root / "sets" / entry["set_id"])
        if result_path is None:
            row["status"] = "not_run"
            rows.append(row)
            continue
        body_result = json.loads(result_path.read_text(encoding="utf-8"))
        prediction_value = body_result.get("outputs", {}).get("fused_body_parameters")
        if not body_result.get("valid") or not prediction_value:
            row.update(
                status=f"body_fuse_{body_result.get('status', 'unknown')}",
                body_fuse_result=str(result_path),
            )
            rows.append(row)
            continue
        prediction = Path(prediction_value)
        if not prediction.is_absolute():
            prediction = (result_path.parent / prediction).resolve()
        metrics = evaluate_fused(reference, prediction)
        measured = {
            "status": "evaluated",
            "body_fuse_status": body_result.get("status"),
            "active_view_count": len(body_result.get("active_view_ids", [])),
            "used_view_count": len(body_result.get("used_view_ids", [])),
            "body_fuse_result": str(result_path),
            **{
                key: metrics[key]
                for key in (
                    "world_mpjpe_mm",
                    "root_aligned_mpjpe_mm",
                    "root_translation_error_mm",
                    "root_rotation_error_deg",
                )
            },
        }
        row.update(measured)
        canonical[entry["set_id"]] = measured
        rows.append(row)
    summary = {
        "schema_version": "camera_rq1_summary_v1",
        "experiment": str(root),
        "reference_sha256": manifest["reference"]["sha256"],
        "evaluated_count": sum(row["status"] in {"evaluated", "duplicate_reused"} for row in rows),
        "pending_count": sum(row["status"] == "not_run" for row in rows),
        "rows": rows,
        "metric_semantics": "round_trip deviation from a frozen MHR estimate, not real-world ground truth",
    }
    destination = Path(output).resolve() if output else root / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "rq1_summary.json", summary)
    keys = sorted({key for row in rows for key in row if key != "camera_ids"}) + ["camera_ids"]
    with (destination / "rq1_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "camera_ids": ";".join(row["camera_ids"])})
    return summary
