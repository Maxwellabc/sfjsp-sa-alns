"""Shipped summaries stay honest: SPM-DAN is N/A, and CP-600s is not invented."""

from __future__ import annotations

import json

import pytest

from sfjsp_alns.cp_fold import FoldError, fold_cp600
from sfjsp_alns.paths import repo_root


def _status(payload: dict) -> str | None:
    if isinstance(payload.get("spmdan"), str):
        return payload["spmdan"]
    if isinstance(payload.get("spmdan_status"), str):
        return payload["spmdan_status"]
    return None


def test_shipped_summaries_keep_spmdan_na():
    root = repo_root() / "results"
    seen = 0
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".rows.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = _status(payload)
        assert status is not None, path.name
        assert status.startswith("N/A"), path.name
        macro = payload.get("macro_E")
        if isinstance(macro, dict) and "SPM-DAN" in macro:
            assert macro["SPM-DAN"] is None
        seen += 1
    assert seen >= 1


def test_shipped_cp60_is_short_budget_not_paper():
    path = repo_root() / "results" / "p0-cp-stoch-summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["protocol"]["time_limit_s"] == 60
    assert payload["n_feasible"] == 20
    assert payload["n_failed"] == 0
    assert payload["spmdan_status"].startswith("N/A")
    assert "3600" in payload["protocol"]["note"]


def test_shipped_k_eval_1000_summary_is_partial():
    path = repo_root() / "results" / "k-eval-1000-summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_instances"] == 20
    assert payload["protocol"]["k_eval"] == 1000
    note = payload["protocol"]["batch_note"]
    assert "20/100" in note
    assert payload["spmdan"].startswith("N/A")


def test_fold_refuses_missing_csv(tmp_path):
    with pytest.raises(FoldError, match="No CP-600s number is invented"):
        fold_cp600(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fold_refuses_short_csv(tmp_path):
    raw = tmp_path / "p0-cp-stoch-600s-raw.csv"
    raw.write_text(
        "instance,status,eval_mean_cmax,eval_var95,cp_obj_search,cp_wall_s,"
        "solution_count,num_realizations_cp,time_limit_s,error\n"
        "10x5+mix_001,FEASIBLE,1,1,1,1,1,10,600,\n",
        encoding="utf-8",
    )
    with pytest.raises(FoldError, match="Refusing to impute"):
        fold_cp600(tmp_path)
    assert not (tmp_path / "p0-cp-stoch-600s-summary.json").exists()
    assert not (tmp_path / "p0-cp-stoch-600s.md").exists()
