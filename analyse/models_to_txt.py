"""把 models/ 全部源码整合为一个 txt 快照，并附上 outputs/ 的指标汇总。

用法：
    python analyse/models_to_txt.py --tag TAR-DCR --run Run1

输出：docs/temporary/models_and_metrics_<tag>_<run>.txt（每次运行新建，不覆盖旧文件）。
"""
import os
import re
import glob
import argparse

from extract_metrics_to_excel import extract_test_block, parse_block, METRIC_KEYS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
OUTPUTS = os.path.join(ROOT, "outputs")
DOCS_TMP = os.path.join(ROOT, "docs", "temporary")

SKIP_DIRS = {"__pycache__", "build", "csrc", ".git"}
SKIP_EXTS = {".pth", ".pt", ".pyc"}


def collect_code():
    files = []
    for root, dirs, names in os.walk(MODELS):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in sorted(names):
            ext = os.path.splitext(n)[1]
            if ext in SKIP_EXTS:
                continue
            if ext in (".py", ".yaml", ".yml"):
                files.append(os.path.relpath(os.path.join(root, n), MODELS))
    return files


def collect_metrics():
    rows = []
    for log in sorted(glob.glob(os.path.join(OUTPUTS, "**", "train_log.txt"), recursive=True)):
        rel = os.path.relpath(os.path.dirname(log), OUTPUTS).replace("\\", "/")
        with open(log, encoding="utf-8", errors="replace") as f:
            block = extract_test_block(f.read())
        if block is None:
            continue
        info = parse_block(block)
        if "F1" in info:
            rows.append((rel, info))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="TAR-DCR")
    ap.add_argument("--run", type=str, default="Run1")
    args = ap.parse_args()

    os.makedirs(DOCS_TMP, exist_ok=True)
    out_path = os.path.join(DOCS_TMP, f"models_and_metrics_{args.tag}_{args.run}.txt")

    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"# STR-RepNet models snapshot + metrics\n")
        out.write(f"# tag={args.tag}  run={args.run}\n")

        files = collect_code()
        out.write(f"\n{'=' * 80}\n## MODELS CODE ({len(files)} files)\n{'=' * 80}\n")
        for rel in files:
            out.write(f"\n{'=' * 80}\n# FILE: models/{rel}\n{'=' * 80}\n")
            with open(os.path.join(MODELS, rel), encoding="utf-8", errors="replace") as f:
                out.write(f.read())

        metrics = collect_metrics()
        out.write(f"\n{'=' * 80}\n## METRICS ({len(metrics)} results)\n{'=' * 80}\n")
        out.write("Run | " + " | ".join(METRIC_KEYS) + " | Params(M) | Trainable(M) | FLOPs(G) | ReparamErr | RepMode\n")
        for rel, info in metrics:
            line = [rel] + [f"{info.get(k):.4f}" if info.get(k) is not None else "-" for k in METRIC_KEYS]
            line += [
                f"{info['Params(M)']:.3f}" if info.get("Params(M)") is not None else "-",
                f"{info['Trainable(M)']:.3f}" if info.get("Trainable(M)") is not None else "-",
                f"{info['FLOPs(G)']:.4f}" if info.get("FLOPs(G)") is not None else "-",
                f"{info['ReparamErr']:.3e}" if info.get("ReparamErr") is not None else "-",
                info.get("RepMode") or "-",
            ]
            out.write(" | ".join(line) + "\n")

    print(f"已写入 {out_path}")
    print(f"  代码文件 {len(files)} 个，指标 {len(metrics)} 行")


if __name__ == "__main__":
    main()
