"""Loader smoke on the shipped synthetic fixtures (no SD3 bank)."""

from __future__ import annotations

from sfjsp_alns.io.fjsp_loader import load_instance
from sfjsp_alns.operators import DESTROY_OPS, REPAIR_OPS
from sfjsp_alns.paths import repo_root


FIXTURES = repo_root() / "data" / "fixtures" / "synthetic"


def test_load_synthetic_fixtures():
    paths = sorted(FIXTURES.glob("*.fjs"))
    assert [p.name for p in paths] == ["tiny_001.fjs", "tiny_002.fjs"]
    for path in paths:
        inst = load_instance(path)
        assert inst.n_jobs == 2
        assert inst.n_machines == 2
        assert inst.n_ops == 4
        assert inst.name == path.stem


def test_operator_registry():
    assert "critical_path" in DESTROY_OPS
    assert "greedy_mean" in REPAIR_OPS
