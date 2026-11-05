#!/usr/bin/env python3
"""Small-scale Stochastic FJSP baseline smoke: generate few realizations + PDR eval.
Does NOT train. No pretrained FJSP-DRL weights ship in the official repo.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path("/workspace/fjsp-combo-b/baselines/NCO-for-Stochastic-FJSP/FJSP-DRL")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Force CPU before params.py / EnvState bind torch.device(configs.device)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = [sys.argv[0], "--device", "cpu", "--device_id", ""]

from common_utils import heuristic_select_action, setup_seed  # noqa: E402
from data_utils import (  # noqa: E402
    load_data_from_single_file,
    matrix_to_text,
    pack_data_from_config,
    pack_data_from_config_realizations,
)
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums as np_env_same  # noqa: E402
from fjsp_env_same_op_nums_torch_no_feat import (  # noqa: E402
    FJSPEnvForSameOpNumsNoFeat as torch_env_same,
)
from generate_random_instances import (  # noqa: E402
    create_stochastic_realizations_random_variances,
)

DATA_SOURCE = "smoke_SD3"
TEST_DATA = "10x5+mix"
VARIANCE = "random"
VARIANCE_DIST = "lognormal"  # empty dist subfolder when lognormal
NUM_SAMPLES = 350  # need >= 250 + num_eval
NUM_EVAL = 50
METHODS = ["FIFO", "MOR", "SPT", "MWKR"]
SEED = 50
OUT_MD = Path("/workspace/fjsp-combo-b/results/stochastic-fjsp-smoke.md")
OUT_NPY = Path("/workspace/fjsp-combo-b/results/stochastic-fjsp-smoke")


def generate_realizations() -> Path:
    src_dir = ROOT / "data" / DATA_SOURCE / TEST_DATA
    dist_str = "" if VARIANCE_DIST == "lognormal" else f"{VARIANCE_DIST}/"
    out_dir = (
        ROOT
        / "data"
        / "stochastic_realizations"
        / VARIANCE
        / f"{dist_str}{DATA_SOURCE}"
        / TEST_DATA
    )
    # fix path join when dist_str empty
    if VARIANCE_DIST == "lognormal":
        out_dir = (
            ROOT
            / "data"
            / "stochastic_realizations"
            / VARIANCE
            / DATA_SOURCE
            / TEST_DATA
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.fjs"))
    print(f"Generating {NUM_SAMPLES} realizations for {len(files)} instances -> {out_dir}")
    for path in tqdm(files, desc="gen"):
        job_length, processing_times = load_data_from_single_file(str(path))
        samples = create_stochastic_realizations_random_variances(
            processing_times, NUM_SAMPLES
        )
        name = path.stem
        realization_file = out_dir / f"{name}.fjs"
        with open(path, "r") as f_or, open(realization_file, "w") as f_real:
            for line in f_or:
                f_real.write(line)
            for i in range(NUM_SAMPLES):
                f_real.write("\n")
                lines_doc = matrix_to_text(job_length, samples[i], "", first_line=False)
                for j in range(len(lines_doc)):
                    f_real.write(lines_doc[j] + "\n")
    return out_dir


def eval_method(data_set, stochastic_realizations, heuristic: str):
    setup_seed(SEED)
    result = []
    for i in tqdm(range(len(data_set[0])), desc=heuristic):
        n_j = data_set[0][i].shape[0]
        n_op, n_m = data_set[1][i].shape
        env = np_env_same(n_j=n_j, n_m=n_m)
        env.set_initial_data([data_set[0][i]], [data_set[1][i]])
        t1 = time.time()
        actions = []
        while True:
            action = heuristic_select_action(heuristic, env)
            actions.append(action)
            _, _, done = env.step(np.array([action]))
            if done:
                break
        t2 = time.time()
        stoch_env = torch_env_same(n_j=n_j, n_m=n_m)
        stoch_env.set_initial_data(
            stochastic_realizations[i][0][0][0][250 : 250 + NUM_EVAL],
            stochastic_realizations[i][0][0][1][250 : 250 + NUM_EVAL],
            group_size=NUM_EVAL,
        )
        for action in actions:
            stoch_env.step(torch.tensor([action]).repeat(NUM_EVAL))
        avg_makespan = float(np.mean(stoch_env.current_makespan.numpy()))
        var_makespan = float(np.percentile(stoch_env.current_makespan.numpy(), 95))
        result.append(
            [
                float(env.current_makespan[0]),
                avg_makespan,
                var_makespan,
                t2 - t1,
            ]
        )
    return np.array(result)


def main():
    log = []
    def L(s=""):
        print(s)
        log.append(s)

    L("# Stochastic FJSP（Smit et al. AAAI 2025）小规模冒烟评测")
    L()
    L(f"- 时间：2026-11-05（Asia/Shanghai）")
    L(f"- 工作区：`{ROOT}`")
    L(f"- Python：{sys.version.split()[0]}；torch={torch.__version__}；numpy={np.__version__}；CUDA={torch.cuda.is_available()}")
    L(f"- 策略：**不训练**；官方 FJSP-DRL 仓未附带预训练权重，故冒烟跑 **PDR 启发式**（论文基线对照：FIFO/MOR/SPT/MWKR），验证场景生成→确定性派工→随机场景回放→期望 Cmax / VaR95 全链路。")
    L()
    L("## 配置")
    L()
    L("| 项 | 值 |")
    L("|---|---|")
    L(f"| data_source | `{DATA_SOURCE}`（自 SD3/10x5+mix 取前 3 例） |")
    L(f"| instances | 3（001–003） |")
    L(f"| variance | `{VARIANCE}` / `{VARIANCE_DIST}` |")
    L(f"| realization samples | {NUM_SAMPLES}（评测切片 `[250:{250+NUM_EVAL}]`） |")
    L(f"| num_eval_realizations | {NUM_EVAL} |")
    L(f"| seed | {SEED} |")
    L(f"| methods | {', '.join(METHODS)} |")
    L()

    t0 = time.time()
    out_dir = generate_realizations()
    L(f"- 场景文件目录：`{out_dir}`")
    L()

    test_data = pack_data_from_config(DATA_SOURCE, [TEST_DATA])
    assert len(test_data) == 1
    data0, name = test_data[0]
    L(f"- 加载实例数：{len(data0[0])}（集名 `{name}`）")

    # load realizations per instance in same sort order as pack_data
    import re

    test_data_path = ROOT / "data" / DATA_SOURCE / TEST_DATA
    instances = sorted(
        sorted(os.listdir(test_data_path), key=lambda s: int(re.findall(r"\d+", s)[0])),
        key=lambda s: int(re.findall(r"\d+", s)[-1]),
    )
    all_stoch = []
    for instance in instances:
        stoch = pack_data_from_config_realizations(
            f"stochastic_realizations/{VARIANCE}/{DATA_SOURCE}/{TEST_DATA}",
            [instance],
            max_realizations=NUM_SAMPLES,
        )
        all_stoch.append(stoch)

    rows = []
    OUT_NPY.mkdir(parents=True, exist_ok=True)
    L()
    L("## 结果（单次 seed；实例级均值）")
    L()
    L("| 方法 | 确定性 Cmax（均值） | 期望 Cmax（场景均值） | VaR95 Cmax | 派工耗时 s（均值） |")
    L("|---|---:|---:|---:|---:|")

    for method in METHODS:
        result = eval_method(data0, all_stoch, method)
        np.save(OUT_NPY / f"Result_{method}_{TEST_DATA}.npy", result)
        det = result[:, 0].mean()
        exp = result[:, 1].mean()
        var = result[:, 2].mean()
        tim = result[:, 3].mean()
        L(f"| {method} | {det:.2f} | {exp:.2f} | {var:.2f} | {tim:.4f} |")
        rows.append((method, det, exp, var, tim, result))

    L()
    L("### 分实例明细（期望 Cmax）")
    L()
    header = "| 实例 | " + " | ".join(METHODS) + " |"
    sep = "|---|" + "---:|" * len(METHODS)
    L(header)
    L(sep)
    for i, inst in enumerate(instances):
        vals = " | ".join(f"{r[5][i,1]:.2f}" for r in rows)
        L(f"| {inst} | {vals} |")

    L()
    L("### VaR95 分实例")
    L()
    L(header)
    L(sep)
    for i, inst in enumerate(instances):
        vals = " | ".join(f"{r[5][i,2]:.2f}" for r in rows)
        L(f"| {inst} | {vals} |")

    elapsed = time.time() - t0
    L()
    L("## 命令与环境")
    L()
    L("```bash")
    L("cd /workspace/fjsp-combo-b")
    L("source .venv/bin/activate")
    L("python scripts/sfjsp_smoke_eval.py")
    L("```")
    L()
    L("## 失败与修复")
    L()
    L("| 问题 | 处理 |")
    L("|---|---|")
    L("| `requirements.txt` 含 `torch==2.3.0+cu121`、`pywin32`、`gurobipy` | 冒烟用 CPU torch + 精简依赖（numpy/scipy/tqdm/pandas），未装 Windows/Gurobi 包 |")
    L("| 官方仓无 FJSP-DRL 预训练权重；评测脚本默认指向未提供的 `test_model` | 改跑论文中的 PDR 对照链路，证明场景协议与指标口径 |")
    L("| `generate_random_instances.py` 用 `split('\\\\')`（Windows） | 冒烟脚本用 `Path.stem` 写 Linux 路径 |")
    L("| 全量 100 例 × 5000 场景过大 | 缩至 3 例 × 350 场景，评测 50 场景 |")
    L("| 无 SPM-DAN 数值 | **非失败**：需作者权重或自行训练；下一步再考虑短训/索取权重 |")
    L()
    L("## 结论")
    L()
    L(f"- **冒烟 PASS**：PDR 全链路在 {elapsed:.1f}s 内跑通，输出确定性 Cmax / 期望 Cmax / VaR95。")
    L("- 数值为 smoke 子集，**不可**直接与论文全文表对比；协议对齐后可扩到 SD3 全量 + 作者模型。")
    L(f"- 原始数组：`{OUT_NPY}/`")
    L()

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
