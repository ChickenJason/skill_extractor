from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUSINESS_MODULES = {"self_annotator", "trf", "instance_discriminator", "aggregator"}


class ArchitectureTests(unittest.TestCase):
    def test_peer_modules_do_not_import_each_other(self) -> None:
        code_root = PROJECT_ROOT / "code"
        violations: list[str] = []
        for module in sorted(BUSINESS_MODULES):
            for path in (code_root / module).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                imported: set[str] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.update(alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported.add(node.module.split(".")[0])
                forbidden = (imported & BUSINESS_MODULES) - {module}
                if forbidden:
                    violations.append(f"{path}: {sorted(forbidden)}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
