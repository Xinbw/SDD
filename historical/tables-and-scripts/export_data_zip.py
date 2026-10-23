"""Rebuild data.zip (the paper's spreadsheet bundle) from the clean-pipeline logs.

data.zip was assembled under the OLD mechanism (local DP + shuffle amplification,
1000 real clients, q=0.1, T=6, per-label eps).  This script regenerates the
sheets whose numbers the new mechanism replaces (notes section 1-2: distributed
discrete Gaussian on the aggregate, 100 clients, simulated cohort 10000, q=1,
T=3, vote + hard-label CE, threshold 0) and copies the rest verbatim.

  acc_compare.xlsx         SSD / DP-FedCT / PATE rebuilt (CIFAR-10 a=0.1, 7 eps x 3
                           seeds); DP-FedDPA / PPFed columns kept from the old file
                           (no valid new runs, notes section 42).  Sheet "MNIST" added.
  acc_compare_claude.xlsx  same, with rho_per_round (zCDP) in place of eps-per-label.
  non-iid.xlsx             CIFAR-10 a=0.1 / a=1.0 rebuilt; a=10 kept (old scheme).
                           Sheet "MNIST" added.  "inc" = gain over the round-1
                           local-only model (mean over seeds).
  comm.xlsx                rebuilt from the T sweep (notes section 38) and the
                           per-round curves of the T=3 main runs.
  hetero_model.xlsx        copied unchanged (no runs under the new mechanism).
  hetero_proxy_data.xlsx   copied unchanged (same).

Usage:  python export_data_zip.py <unpacked old data.zip dir> <out dir>
"""

import glob
import re
import shutil
import statistics as st
import sys
import zipfile

import openpyxl
from openpyxl.styles import Alignment, Font

from dp_utils import eps_to_zcdp
from export_comparison import (CIFAR_A1_K, CIFAR_K, EPS, MNIST_A1_K, MNIST_K,
                               a1_dir, cifar_fedct, cifar_pate, cifar_ssd,
                               mnist_fedct, mnist_pate, mnist_ssd)

SEEDS = (42, 43, 44)
T = 3
DELTA = 1e-5
CFG = ("client=100 (cohort 10000 simulated), q=1, proxy_num=2000, round=3, "
       "delta=1e-5, vote + hard-label CE, thr=0, 3 seeds")

MEAN = re.compile(r"Final mean (?:client )?accuracy: ([\d.]+)")
BEST = re.compile(r"Best  client: id=\d+, accuracy=([\d.]+)")
WORST = re.compile(r"Worst client: id=\d+, accuracy=([\d.]+)")
ROUND = re.compile(r"^\[Round (\d+)\] valid_acc=([\d.]+)(?:\s+std=[\d.]+\s+best=([\d.]+)\s+worst=([\d.]+))?",
                   re.M)


def parse(path):
    t = open(path, errors="ignore").read()
    m = MEAN.search(t)
    if not m:
        return None
    rounds = {int(r): (float(a), b, w) for r, a, b, w in ROUND.findall(t)}
    last = rounds[max(rounds)]
    best = BEST.search(t)
    worst = WORST.search(t)
    return {
        "avg": float(m.group(1)),
        "max": float(best.group(1)) if best else float(last[1]),
        "min": float(worst.group(1)) if worst else float(last[2]),
        "r1": rounds[1][0],
        "rounds": [rounds[k][0] for k in sorted(rounds)],
    }


def first(patterns):
    for p in patterns:
        for h in sorted(glob.glob(p)):
            v = parse(h)
            if v is not None:
                return v
    return None


def series(e, K, fn):
    return [v for v in (first(fn(e, K, s)) for s in SEEDS) if v is not None]


def agg(vals):
    """acc_min / acc_avg / acc_max / min_std / avg_std over seeds (None if n<2)."""
    n = len(vals)
    mn = [v["min"] for v in vals]
    av = [v["avg"] for v in vals]
    mx = [v["max"] for v in vals]
    return dict(n=n, acc_min=st.mean(mn), acc_avg=st.mean(av), acc_max=st.mean(mx),
                min_std=st.stdev(mn) if n > 1 else None,
                avg_std=st.stdev(av) if n > 1 else None,
                r1=st.mean(v["r1"] for v in vals))


# ---------------------------------------------------------------- sheet helpers
BOLD = Font(bold=True)
CENTER = Alignment(horizontal="center")


def put(ws, r, c, v, bold=False):
    cell = ws.cell(row=r, column=c, value=v)
    if bold:
        cell.font = BOLD
    return cell


def block(ws, r0, c0, title, cols, rows, label, note=None):
    """Old-file layout: merged config title, header, rows, merged label at the bottom."""
    w = len(cols)
    ws.merge_cells(start_row=r0, start_column=c0, end_row=r0, end_column=c0 + w - 1)
    put(ws, r0, c0, title)
    for j, h in enumerate(cols):
        put(ws, r0 + 1, c0 + j, h, bold=True)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            put(ws, r0 + 2 + i, c0 + j, v)
    rl = r0 + 2 + len(rows)
    ws.merge_cells(start_row=rl, start_column=c0, end_row=rl, end_column=c0 + w - 1)
    put(ws, rl, c0, label, bold=True).alignment = CENTER
    if note:
        ws.merge_cells(start_row=rl + 1, start_column=c0, end_row=rl + 1, end_column=c0 + w - 1)
        put(ws, rl + 1, c0, note)
    return rl + (2 if note else 1)


def method_rows(kmap, fn, with_K=False, with_rho=False):
    rows = []
    for e in EPS:
        K = kmap[e]
        a = agg(series(e, K, fn))
        row = []
        if with_rho:
            row.append(float(f"{eps_to_zcdp(float(e), DELTA) / T:.3g}"))
        row += [float(e), round(a["acc_min"], 4), round(a["acc_avg"], 4), round(a["acc_max"], 4),
                None if a["min_std"] is None else round(a["min_std"], 4),
                None if a["avg_std"] is None else round(a["avg_std"], 4)]
        if with_K:
            row.append(K)
        row.append(a["n"])
        rows.append(row)
    return rows


def copy_block(src_ws, r0, c0, r1, c1, dst_ws, dr0, dc0):
    """Copy a rectangular block of values (+ merges inside it) between sheets."""
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            v = src_ws.cell(row=r, column=c).value
            if v is not None:
                put(dst_ws, dr0 + r - r0, dc0 + c - c0, v)
    for mr in list(src_ws.merged_cells.ranges):
        if r0 <= mr.min_row and mr.max_row <= r1 and c0 <= mr.min_col and mr.max_col <= c1:
            dst_ws.merge_cells(start_row=dr0 + mr.min_row - r0, start_column=dc0 + mr.min_col - c0,
                               end_row=dr0 + mr.max_row - r0, end_column=dc0 + mr.max_col - c0)


# ---------------------------------------------------------------- acc_compare
def build_acc_compare(old_path, out_path, with_rho):
    old = openpyxl.load_workbook(old_path, data_only=True)["Sheet1"]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.merge_cells("A1:AD1")
    put(ws, 1, 1, "准确率测试 —— CIFAR-10, alpha=0.1（新方案：聚合上的分布式离散高斯；"
                  "由 export_data_zip.py 从日志生成）", bold=True)
    cols = ["epsilon_total", "acc_min", "acc_avg", "acc_max", "min_std", "avg_std", "seeds"]
    c = 1
    if with_rho:
        cols = ["rho_per_round"] + cols
    end = block(ws, 2, c, CFG + ", K=10", cols,
                method_rows(CIFAR_K, cifar_ssd, with_rho=with_rho), "ssd")
    c += len(cols) + 1
    bcols = ["epsilon_total", "acc_min", "acc_avg", "acc_max", "min_std", "avg_std", "seeds"]
    block(ws, 2, c, CFG + ", central DP sqrt(2N)", bcols, method_rows(CIFAR_K, cifar_fedct), "DP-fedct")
    c += len(bcols) + 1
    block(ws, 2, c, CFG + ", data-independent accounting (lower bound)", bcols,
          method_rows(CIFAR_K, cifar_pate), "PATE")
    c += len(bcols) + 1
    # old DP-FedDPA / PPFed blocks: rows 2..12, columns T..Y (20..25) and Z..AE (26..31)
    for (oc0, oc1, lab) in ((20, 25, "DP-FedDPA"), (26, 31, "PPFed")):
        rows = []
        for r in range(4, 12):
            vals = [old.cell(row=r, column=cc).value for cc in range(oc0, oc1 + 1)]
            if vals[0] is not None:
                rows.append(vals)
        hdr = [old.cell(row=3, column=cc).value for cc in range(oc0, oc1 + 1)]
        block(ws, 2, c, str(old.cell(row=2, column=oc0).value), hdr, rows, lab,
              note="旧方案数据，未更新（新方案下无有效 run，见 notes §42）")
        c += len(hdr) + 1
    for col in range(1, c + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 13

    # second sheet: the other three combinations
    ws2 = wb.create_sheet("MNIST_and_alpha1")
    r = 1
    for title, kmap, fns in (
            ("MNIST, alpha=0.1", MNIST_K, (mnist_ssd, mnist_fedct, mnist_pate)),
            ("CIFAR-10, alpha=1.0", CIFAR_A1_K, a1_dir("alpha1")),
            ("MNIST, alpha=1.0", MNIST_A1_K, a1_dir("alpha1_mnist"))):
        put(ws2, r, 1, title + " —— " + CFG, bold=True)
        r += 1
        c = 1
        ends = []
        for lab, fn, with_K in (("ssd", fns[0], True), ("DP-fedct", fns[1], False), ("PATE", fns[2], False)):
            cols = (["epsilon_total", "acc_min", "acc_avg", "acc_max", "min_std", "avg_std"]
                    + (["K"] if with_K else []) + ["seeds"])
            ends.append(block(ws2, r, c, title, cols, method_rows(kmap, fn, with_K=with_K), lab))
            c += len(cols) + 1
        r = max(ends) + 1
    for col in range(1, 30):
        ws2.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 13
    wb.save(out_path)


# ---------------------------------------------------------------- non-iid
def noniid_rows(kmap, fn):
    rows = []
    for e in EPS:
        a = agg(series(e, kmap[e], fn))
        rows.append([float(e), round(a["acc_min"], 4), round(a["acc_avg"], 4), round(a["acc_max"], 4),
                     round(a["acc_min"] - a["r1"], 4), round(a["acc_avg"] - a["r1"], 4),
                     round(a["r1"], 4), kmap[e], a["n"]])
    return rows


def build_noniid(old_path, out_path):
    old = openpyxl.load_workbook(old_path, data_only=True)["Sheet1"]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    cols = ["epsilon_total", "acc_min", "acc_avg", "acc_max", "min_inc", "avg_inc", "local_r1", "K", "seeds"]
    note = "inc = 相对 round-1 本地模型（协作前）的均值提升；local_r1 为该基线"
    c = 1
    block(ws, 1, c, "SSD, CIFAR-10, alpha=0.1 —— " + CFG, cols, noniid_rows(CIFAR_K, cifar_ssd),
          "0.1-cifar10", note=note)
    c += len(cols) + 1
    block(ws, 1, c, "SSD, CIFAR-10, alpha=1.0 —— " + CFG, cols,
          noniid_rows(CIFAR_A1_K, a1_dir("alpha1")[0]), "10-cifar10".replace("10-", "1-"), note=note)
    c += len(cols) + 1
    # alpha=10 block from the old file: rows 1..11, columns O..U (15..21)
    copy_block(old, 1, 15, 11, 21, ws, 1, c)
    ws.merge_cells(start_row=12, start_column=c, end_row=12, end_column=c + 6)
    put(ws, 12, c, "旧方案数据，未更新（新方案未跑 alpha=10）")
    for col in range(1, 32):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 13

    ws2 = wb.create_sheet("MNIST")
    c = 1
    block(ws2, 1, c, "SSD, MNIST, alpha=0.1 —— " + CFG, cols, noniid_rows(MNIST_K, mnist_ssd),
          "0.1-mnist", note=note)
    c += len(cols) + 1
    block(ws2, 1, c, "SSD, MNIST, alpha=1.0 —— " + CFG, cols,
          noniid_rows(MNIST_A1_K, a1_dir("alpha1_mnist")[0]), "1-mnist", note=note)
    for col in range(1, 22):
        ws2.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 13
    wb.save(out_path)


# ---------------------------------------------------------------- comm (rounds)
def rounds_run(ds, e, Tn, s):
    tag = e.replace(".", "_")
    return parse(f"logs/dp_v2/rounds/{ds}_eps{tag}_T{Tn}_s{s}.log") if glob.glob(
        f"logs/dp_v2/rounds/{ds}_eps{tag}_T{Tn}_s{s}.log") else None


def build_comm(out_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    # block A: T sweep (separate runs; sigma grows with sqrt(T) at fixed eps_total)
    rows = []
    for Tn in (2, 3, 5, 8):
        row = [Tn]
        for ds, e in (("cifar", "0.1"), ("cifar", "1.0"), ("cifar", "1000"), ("mnist", "0.1")):
            if Tn == 3 and e != "1000":
                K = {"cifar": CIFAR_K, "mnist": MNIST_K}[ds][e]
                fn = {"cifar": cifar_ssd, "mnist": mnist_ssd}[ds]
                vals = [v["avg"] for v in series(e, K, fn)]
                row.append(round(st.mean(vals), 4))
                continue
            vals = [v["avg"] for v in (rounds_run(ds, e, Tn, s) for s in SEEDS) if v is not None]
            row.append(round(st.mean(vals), 4) if vals else "无记录")
        rows.append(row)
    cols = ["T (rounds)", "cifar eps=0.1", "cifar eps=1.0", "cifar no-privacy (eps=1000)", "mnist eps=0.1"]
    end = block(ws, 1, 1, "SSD 轮数扫描（notes §38）：固定 eps_total，sigma ∝ sqrt(T)；"
                          "T=3 与 T=5 为 3 seed 均值，其余 seed 42；CIFAR K=10、MNIST K=50",
                cols, rows, "ssd")
    # block B: per-round accuracy inside the T=3 main runs, eps=1.0 and 0.1, 3-seed mean
    c = len(cols) + 2
    for e in ("1.0", "0.1"):
        K = CIFAR_K[e]
        curves = {lab: [v["rounds"] for v in series(e, K, fn)]
                  for lab, fn in (("ssd", cifar_ssd), ("fedct", cifar_fedct), ("dp-pate", cifar_pate))}
        rows = []
        for r in range(T):
            rows.append([r + 1] + [round(st.mean(cv[r] for cv in curves[lab]), 4)
                                   for lab in ("ssd", "fedct", "dp-pate")])
        block(ws, 1, c, f"CIFAR-10 alpha=0.1, eps={e}, T=3 主线 run 的逐轮准确率（3 seed 均值；"
                        f"round 1 = 协作前本地模型）",
              ["round", "ssd", "fedct", "dp-pate"], rows, f"eps={e}")
        c += 5
    for col in range(1, 20):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 15
    wb.save(out_path)


README = """data.zip —— 论文用表格（2026-09-22 重建）

新方案 = 聚合上的分布式离散高斯 + 多消息洗牌（SDD_PHASE1_NOTES.md §1），
100 客户端、cohort 10000（模拟）、q=1、T=3、K 逐 ε 取值、vote + 硬标签 CE、阈值 0、3 seeds。
旧方案 = 本地 DP + 洗牌放大，1000 客户端、q=0.1、T=6、按标签 ε。

| 文件 | 状态 |
|---|---|
| acc_compare.xlsx | Sheet1：SSD / DP-FedCT / PATE 已换成新方案（CIFAR-10 α=0.1）；DP-FedDPA / PPFed 保留旧方案数据。新增 sheet MNIST_and_alpha1（另三个组合） |
| acc_compare_claude.xlsx | 同上，首列 eps-per-label 改为 rho_per_round（zCDP，新方案无按标签 ε） |
| non-iid.xlsx | α=0.1 / α=1.0 已换新方案；α=10 保留旧方案。inc 改为相对 round-1 本地模型。新增 sheet MNIST |
| comm.xlsx | 重建：T 扫描（σ ∝ √T）+ T=3 主线 run 的逐轮曲线 |
| hetero_model.xlsx | 未改（新方案没跑异构架构） |
| hetero_proxy_data.xlsx | 未改（新方案没跑 CIFAR-100 / EMNIST 代理） |

生成脚本：export_data_zip.py。数字与 SSD_RESULTS.md / SSD_COMPARISON.md 一致。
"""


def main():
    src, out = sys.argv[1], sys.argv[2]
    import os
    os.makedirs(out, exist_ok=True)
    build_acc_compare(f"{src}/acc_compare.xlsx", f"{out}/acc_compare.xlsx", with_rho=False)
    build_acc_compare(f"{src}/acc_compare_claude.xlsx", f"{out}/acc_compare_claude.xlsx", with_rho=True)
    build_noniid(f"{src}/non-iid.xlsx", f"{out}/non-iid.xlsx")
    build_comm(f"{out}/comm.xlsx")
    for f in ("hetero_model.xlsx", "hetero_proxy_data.xlsx"):
        shutil.copy(f"{src}/{f}", f"{out}/{f}")
    open(f"{out}/README.txt", "w").write(README)
    with zipfile.ZipFile("data.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out)):
            z.write(f"{out}/{f}", f)
    print("wrote data.zip:", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
