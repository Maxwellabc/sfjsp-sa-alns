"""`.fjs` codec round-trip on the shipped fixtures."""

from __future__ import annotations

import numpy as np

from sfjsp_alns.io.fjs_text import matrix_to_text, read_instance_text, text_to_matrix
from sfjsp_alns.paths import repo_root


def test_fixture_roundtrip():
    root = repo_root() / "data" / "fixtures" / "synthetic"
    for path in sorted(root.glob("*.fjs")):
        job_length, op_pt = read_instance_text(path)
        text = matrix_to_text(job_length, op_pt, op_per_mch=1.0)
        job2, op2 = text_to_matrix(text)
        assert np.array_equal(job_length, job2)
        assert np.array_equal(op_pt, op2)
        assert int(op_pt.shape[0]) == int(job_length.sum())
