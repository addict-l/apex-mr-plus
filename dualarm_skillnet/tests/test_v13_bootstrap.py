"""Offline checks for the isolated simulator integration."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_skillnet_uses_only_standard_library(self):
        allowed = sys.stdlib_module_names | {"skillnet", "__future__"}
        for path in (ROOT / "skillnet").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    self.assertIn(module.split(".")[0], allowed,
                                  f"Third-party import in {path}: {module}")

    def test_local_execution_stack_has_provenance(self):
        records = json.loads((ROOT / "apexmr/PROVENANCE.json").read_text(
            encoding="utf-8"))["files"]
        names = {record["file"] for record in records}
        for path in (ROOT / "apexmr").iterdir():
            if path.suffix not in {".py", ".lua"} or path.name == "__init__.py":
                continue
            self.assertIn(path.name, names)
        for record in records:
            path = ROOT / "apexmr" / record["file"]
            self.assertTrue(path.is_file())
            self.assertRegex(record["source_sha256"], r"^[a-f0-9]{64}$")
            self.assertTrue(Path(record["source"]).is_absolute())
            if not record["modifications"]:
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 record["source_sha256"], record["file"])


if __name__ == "__main__":
    unittest.main()
