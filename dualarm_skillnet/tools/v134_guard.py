"""v134 protection checks; v133 artifacts and forbidden source stay immutable."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old

BASE = ROOT / "results/sim_v1_3/v134"
AUTHORIZED = frozenset({"skillnet/planning/search.py", "skillnet/planning/heuristic.py",
    "skillnet/planning/bounds.py", "skillnet/planning/allocation.py", "skillnet/pipeline.py",
    "simbridge/backend.py", "simbridge/qualifier.py", "simbridge/taskio.py", "simbridge/__main__.py"})


class Guard:
    def __init__(self, baseline=None, *, project=ROOT, base=BASE, source_base=None, archive_guard=None):
        self.project, self.base = Path(project).resolve(), Path(base).resolve()
        self.source_base = Path(source_base or self.project / "results/sim_v1_3/v133").resolve()
        self.baseline = baseline if baseline is not None else json.loads((self.base / "baseline.json").read_text(encoding="utf-8"))
        self.archive_guard = archive_guard if archive_guard is not None else old.ArchiveGuard()
        self.cache = {}

    def _sha(self, path):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.cache:
            self.cache[key] = old.sha256(path)
        return self.cache[key]

    def check(self):
        forbidden, authorized, changed_old = [], [], []
        for name, expected in self.baseline.get("project_files", {}).items():
            path = self.project / name
            if not path.is_file() or self._sha(path) != expected:
                if name in AUTHORIZED or name == "README.md":
                    authorized.append(name)
                else:
                    forbidden.append(name)
        expected_old = self.baseline.get("v133_files", {})
        actual_old = {p.relative_to(self.project).as_posix(): p for p in self.source_base.rglob("*") if p.is_file()}
        for name in sorted(set(expected_old) | set(actual_old)):
            if name not in expected_old or name not in actual_old or self._sha(actual_old[name]) != expected_old[name]:
                changed_old.append(name)
        baseline_core = {name for name in self.baseline.get("project_files", {}) if name.startswith("skillnet/") and name.endswith(".py")}
        current_core = {p.relative_to(self.project).as_posix() for p in (self.project / "skillnet").rglob("*.py")}
        unexpected_core = sorted(current_core - baseline_core - {"skillnet/planning/allocation.py"})
        archive = self.archive_guard.check()
        return {"passed": not forbidden and not changed_old and not unexpected_core and archive.get("passed") is True,
            "checked_at": old.utc_now(), "protected_project_files": len(self.baseline.get("project_files", {})),
            "protected_v133_files": len(expected_old), "forbidden_source_changes": forbidden,
            "authorized_changes": authorized, "v133_changes": changed_old,
            "unexpected_new_core_files": unexpected_core, "archives": archive,
            "authorization": "FIX_v1.3.4; original accepted external archive baseline retained"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="v134源码/历史结果/旧归档保护检查")
    parser.add_argument("--out", type=Path, default=BASE / "archive_check.json")
    args = parser.parse_args(argv)
    if not args.out.resolve().is_relative_to(BASE.resolve()):
        raise ValueError("校验结果只可写v134")
    result = Guard().check()
    old.atomic_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
