from __future__ import annotations

from pathlib import Path

import nbformat


def test_notebook_is_valid_and_uses_package_modules():
    notebook_path = Path("jobguard-classifier.ipynb")
    nb = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(nb)

    code_cells = ["".join(cell.get("source", [])) for cell in nb.cells if cell.get("cell_type") == "code"]
    assert code_cells, "Expected the notebook to contain code cells"

    for idx, source in enumerate(code_cells, start=1):
        assert source.strip(), f"Code cell {idx} should not be empty"
        compile(source, f"notebook_cell_{idx}", "exec")

    joined = "\n".join(code_cells)
    assert "from jobguard.pipeline import" in joined
    assert "from jobguard.detector import JobFraudDetector" in joined
    assert "df = build_training_frame(df)" in joined
    assert "JobFraudDetector.from_artifacts(MODEL_DIR)" in joined

