"""Validate the public package without importing attack dependencies."""

import ast
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [p for group in ("attacks", "ablations", "evaluation")
               for p in sorted((root / group).glob("*.py"))]
    assert len(scripts) == 21, f"Expected 21 experiment scripts, found {len(scripts)}"
    inventory = {row["file"]: row for row in json.loads(
        (root / "docs" / "CODE_INVENTORY.json").read_text(encoding="utf-8"))}
    placeholders = 0
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"[\u3400-\u9fff]", text), f"Non-English text in {path.name}"
        compile(text, str(path), "exec")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                assert not re.match(r"^[A-Za-z]:[\\/]", value), path.name
                assert not value.startswith(("/home/", "/Users/", "\\\\")), path.name
                if value == "......":
                    placeholders += 1
        relative = path.relative_to(root).as_posix()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == inventory[relative]["public_sha256"], (
            f"Inventory hash differs for {relative}; update it after editing the script."
        )
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_check_paths_configured")
        namespace = {"MODEL_PATH": "......"}
        module = ast.Module(body=[helper], type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
        try:
            namespace["_check_paths_configured"]()
        except ValueError as error:
            assert "MODEL_PATH" in str(error)
        else:
            raise AssertionError(f"Missing unresolved-path guard in {path.name}")
        namespace["MODEL_PATH"] = "checkpoint.pth"
        namespace["MODEL_SPECS"] = [{"weight_path": "......"}]
        try:
            namespace["_check_paths_configured"]()
        except ValueError as error:
            assert "weight_path" in str(error)
        else:
            raise AssertionError("Nested checkpoint placeholders must be rejected")
        namespace["MODEL_SPECS"] = [{"weight_path": "checkpoint.pth"}]
        namespace["_check_paths_configured"]()
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "![MASS method overview](method.PNG)" in readme
    assert (root / "method.PNG").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for file in root.rglob("*.py"):
        assert not re.search(r"[\u3400-\u9fff]", file.read_text(encoding="utf-8"))
    print(f"PASS: {len(scripts)} experiment scripts; syntax and path guards verified.")
    print(f"PASS: {placeholders} sanitized path fields; inventory hashes match.")
    print("PASS: root README embeds method.PNG; no attack dependencies were imported.")


if __name__ == "__main__":
    main()
