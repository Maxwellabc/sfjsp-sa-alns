"""Smit FJSP-DRL ``.fjs`` text codec (no dependency on the NCO checkout).

Matches ``text_to_matrix`` / ``matrix_to_text`` in
``NCO-for-Stochastic-FJSP/FJSP-DRL/data_utils.py``: machines are 1-indexed in
the text form and 0-indexed in ``op_pt``. A zero entry means incompatible.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_NUM = re.compile(r"\d+\.?\d*")


def matrix_to_text(
    job_length: np.ndarray,
    op_pt: np.ndarray,
    op_per_mch: object = "",
    first_line: bool = True,
) -> list[str]:
    job_length = np.asarray(job_length)
    op_pt = np.asarray(op_pt)
    n_j = int(job_length.shape[0])
    n_m = int(op_pt.shape[1])
    text: list[str] = []
    if first_line:
        text.append(f"{n_j}\t{n_m}\t{op_per_mch}")
    op_idx = 0
    for j in range(n_j):
        n_ops = int(job_length[j])
        line = f"{n_ops}"
        for _ in range(n_ops):
            use_mch = np.flatnonzero(op_pt[op_idx] != 0)
            line += " " + str(int(use_mch.shape[0]))
            for k in use_mch:
                line += f" {int(k) + 1} {int(op_pt[op_idx, k])}"
            op_idx += 1
        text.append(line)
    if op_idx != int(op_pt.shape[0]):
        raise ValueError("job_length does not sum to the operation count")
    return text


def text_to_matrix(text: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if len(text) < 2:
        raise ValueError("FJSP text needs a header and at least one job line")
    header_nums = _NUM.findall(text[0])
    if len(header_nums) < 2:
        raise ValueError(f"bad FJSP header: {text[0]!r}")
    n_j = int(float(header_nums[0]))
    n_m = int(float(header_nums[1]))
    if len(text) < n_j + 1:
        raise ValueError(f"expected {n_j} job lines, got {len(text) - 1}")
    job_length = np.zeros(n_j, dtype=np.int64)
    op_pt_rows: list[np.ndarray] = []
    for i in range(n_j):
        content = [int(float(s)) for s in _NUM.findall(text[i + 1])]
        if not content:
            raise ValueError(f"empty job line {i}")
        job_length[i] = content[0]
        idx = 1
        for _ in range(int(content[0])):
            if idx >= len(content):
                raise ValueError(f"truncated job line {i}")
            mch_num = content[idx]
            row = np.zeros(n_m, dtype=np.int64)
            for k in range(mch_num):
                mch_id = content[idx + 2 * k + 1]
                pt = content[idx + 2 * k + 2]
                if mch_id < 1 or mch_id > n_m:
                    raise ValueError(f"machine id {mch_id} outside 1..{n_m}")
                row[mch_id - 1] = pt
            idx = idx + 2 * mch_num + 1
            op_pt_rows.append(row)
    if not op_pt_rows:
        raise ValueError("instance has no operations")
    return job_length, np.stack(op_pt_rows, axis=0)


def read_instance_text(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in raw.split("\n") if ln.strip() != ""]
    return text_to_matrix(lines)


def write_stochastic_realizations_file(
    path: str | Path,
    job_length: np.ndarray,
    nominal: np.ndarray,
    samples: np.ndarray,
    op_per_mch: object = "",
) -> None:
    """Nominal block, then each sample separated by a blank line.

    This is the layout ``load_data_from_stochastic_realizations_file`` reads:
    the first block is deterministic and is **not** scenario index 0; index 0
    is the first sample after the blank line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples)
    if samples.ndim != 3:
        raise ValueError("samples must have shape (K, N, M)")
    header = matrix_to_text(job_length, nominal, op_per_mch, first_line=True)
    parts = ["\n".join(header) + "\n"]
    for i in range(int(samples.shape[0])):
        body = matrix_to_text(job_length, samples[i], "", first_line=False)
        parts.append("\n".join(body) + "\n")
    text = parts[0]
    for chunk in parts[1:]:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + chunk
    path.write_text(text, encoding="utf-8")


def read_realization_matrices(path: str | Path, max_realizations: int = 1100) -> list[np.ndarray]:
    """Load stochastic blocks the way FJSP-DRL does.

    ``max_realizations`` follows the official off-by-one: at most
    ``max_realizations + 1`` blocks are returned. Pass a value ``>= K - 1``
    when ``K`` scenarios are required (the K_eval=1000 slice ``[250:1250]``
    needs ``max_realizations >= 1249``).
    """
    raw = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if "\n\n" not in raw:
        raise ValueError(
            f"{path} has no blank-line separators; expected a Smit realizations file "
            "(deterministic block, blank line, then one block per scenario)"
        )
    head, rest = raw.split("\n\n", 1)
    first_line = head.split("\n", 1)[0]
    if not first_line.strip():
        raise ValueError(f"{path} is missing the instance header")
    blocks = rest.split("\n\n")
    out: list[np.ndarray] = []
    limit = int(max_realizations) + 1
    for block in blocks[:limit]:
        lines = [first_line] + [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        _, op_pt = text_to_matrix(lines)
        out.append(np.asarray(op_pt, dtype=np.int64))
    if not out:
        raise ValueError(f"no stochastic realizations in {path}")
    return out
