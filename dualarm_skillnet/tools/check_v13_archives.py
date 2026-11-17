"""Read-only verification of the v1.3 archive and protected-code baselines."""
from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/sim_v1_3"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> int:
    started = time.monotonic()
    before = json.loads((OUTPUT / "archive_before.json").read_text(encoding="utf-8"))
    report = {"algorithm": "SHA256", "archives": {}, "protected_changes": []}
    for name, archive in before["archives"].items():
        directory = Path(archive["root"])
        expected = {item["path"]: item for item in archive["files"]}
        current = {path.relative_to(directory).as_posix(): path
                   for path in directory.rglob("*") if path.is_file()}
        changes = []
        for relative in sorted(expected.keys() | current.keys()):
            if relative not in expected:
                changes.append({"path": relative, "change": "added"})
            elif relative not in current:
                changes.append({"path": relative, "change": "removed"})
            else:
                path = current[relative]
                item = expected[relative]
                if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
                    changes.append({"path": relative, "change": "modified"})
        report["archives"][name] = {"file_count": len(current), "changes": changes}
    protected = json.loads((OUTPUT / "protected_before.json").read_text(encoding="utf-8"))
    revision_baseline = OUTPUT / "v131_protected_before.json"
    if revision_baseline.is_file():
        protected.update(json.loads(revision_baseline.read_text(encoding="utf-8")))
    status_baseline = OUTPUT / "v132_status_protected_before.json"
    report["accepted_method_changes"] = []
    if status_baseline.is_file():
        accepted = json.loads(status_baseline.read_text(encoding="utf-8"))
        protected.update(accepted["protected_files"])
        for relative, saved in accepted["accepted_method_ast"].items():
            tree = ast.parse((ROOT/relative).read_text(encoding="utf-8-sig"))
            current = {node.name: ast.dump(node, include_attributes=False)
                       for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name in saved}
            report["accepted_method_changes"].extend(
                relative+":"+name for name, expected in saved.items()
                if current.get(name) != expected)
    report["protected_file_count"] = len(protected)
    for relative, expected_hash in protected.items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected_hash:
            report["protected_changes"].append(relative)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["passed"] = (not report["protected_changes"] and not report["accepted_method_changes"] and
                        all(not item["changes"] for item in report["archives"].values()))
    (OUTPUT / "archive_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
