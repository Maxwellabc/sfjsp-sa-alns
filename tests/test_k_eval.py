"""Synthetic K_eval smoke. Does not record or assert paper makespans."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from sfjsp_alns.k_eval import KEvalSpec, SPMDAN_STATUS, assert_slices_disjoint
from sfjsp_alns.paths import repo_root


def test_overlapping_slices_refused():
    spec = KEvalSpec(k_eval=10, eval_start=0, k_search=10, search_start=0)
    with pytest.raises(ValueError, match="overlaps"):
        assert_slices_disjoint(spec)


def test_cli_two_instance_alns_smoke(tmp_path):
    """2 fixtures, 3 ALNS iterations, held-out synthetic slices. SPM-DAN stays N/A."""
    out = tmp_path / "k-eval-smoke.json"
    script = repo_root() / "scripts" / "alns_k_eval_1000.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root() / "src")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic",
            "--instances",
            "2",
            "--k-eval",
            "4",
            "--eval-start",
            "8",
            "--k-search",
            "4",
            "--max-iter",
            "3",
            "--time-limit",
            "2",
            "--seed",
            "50",
            "--out",
            str(out),
        ],
        cwd=repo_root(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_instances"] == 2
    assert data["protocol"]["kind"] == "synthetic_fixture"
    assert data["protocol"]["max_iter"] == 3
    assert data["protocol"]["comparable_to_paper_protocol"] is False
    assert data["spmdan"] == SPMDAN_STATUS
    assert data["spmdan"].startswith("N/A")
    assert set(data["macro_mean_cmax"]) == {"FIFO", "MOR", "SPT", "MWKR", "SA-ALNS"}
    for value in data["macro_mean_cmax"].values():
        assert value > 0
    assert "SA-ALNS" in out.with_suffix(".md").read_text(encoding="utf-8")
