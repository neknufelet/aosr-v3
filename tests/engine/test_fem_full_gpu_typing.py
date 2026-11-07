"""選用 nvmath 不必安裝，實際工廠的每個呼叫仍須受嚴格型別檢查。"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_nvmath_factory_rejects_invalid_calls_with_mypy(tmp_path: Path) -> None:
    """退回 ModuleType 或把參數放寬時，拼錯與錯型別不再報錯，這題必紅。"""
    tree = ast.parse((REPO / "src/aosr/physics/fem_full_gpu.py").read_text(encoding="utf-8"))
    # 保留實際工廠及它的宣告，不執行 import；不帶入無關的完整解修正迴圈。
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.ClassDef) or node.name != "GPUFullSolver"]
    correct = ast.unparse(tree)
    probe = tmp_path / "nvmath_probe.py"
    command = ["mypy", "--config-file", str(REPO / "pyproject.toml"),
               "--cache-dir", str(tmp_path / "mypy-cache"), "--show-error-codes",
               "--no-color-output", str(probe)]
    probe.write_text(correct, encoding="utf-8")
    clean = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    mistakes = [
        ("advanced.DirectSolver(", "advanced.DirectSovler(", "[attr-defined]"),
        ("options=", "wrong_options=", "[call-arg]"),
        ("sparse_system_type=", "wrong_system_type=", "[call-arg]"),
        ("sparse_system_type=advanced.DirectSolverMatrixType.SYMMETRIC",
         "sparse_system_type='symmetric'", "[arg-type]"),
        ("advanced.DirectSolver(a, b,", "advanced.DirectSolver(a, 'bad_rhs',", "[arg-type]"),
        ("advanced.DirectSolver(a, b,", "advanced.DirectSolver('bad_matrix', b,", "[arg-type]"),
        ("options=advanced.DirectSolverOptions(sparse_system_type=advanced.DirectSolverMatrixType.SYMMETRIC)",
         "options='bad_options'", "[arg-type]"),
        ("DirectSolverMatrixType.SYMMETRIC", "DirectSolverMatrixType.SYMETRIC", "[attr-defined]"),
    ]
    # 每個反例各用一個新檔名：同一路徑在同一秒內覆寫，mypy 快取可能沿用正例結果而假紅。
    for index, (original, replacement, diagnostic) in enumerate(mistakes):
        assert original in correct
        wrong_probe = tmp_path / f"nvmath_probe_wrong_{index}.py"
        wrong_probe.write_text(replacement.join(correct.split(original)), encoding="utf-8")
        wrong = subprocess.run([*command[:-1], str(wrong_probe)], cwd=REPO, capture_output=True, text=True,
                               check=False)
        output = wrong.stdout + wrong.stderr
        assert wrong.returncode == 1, f"{replacement}: mypy accepted an invalid call\n{output}"
        assert diagnostic in output, output
