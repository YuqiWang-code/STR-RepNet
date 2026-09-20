# -*- coding: utf-8 -*-
"""把 docs/参考文献 目录下的所有 PDF 文件整合为一个 txt 文件。

用法：
    python merge_pdfs_to_txt.py                # 使用默认目录与输出文件名
    python merge_pdfs_to_txt.py <pdf目录> <输出txt路径>   # 可选覆盖

输出：默认生成 docs/参考文献/参考文献合集.txt
"""

import sys
from pathlib import Path

# 项目根目录（本文件位于 analyse/ 下，根目录即其上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认的 PDF 目录与输出文件
PDF_DIR = PROJECT_ROOT / "docs" / "参考文献"
OUTPUT_TXT = PDF_DIR / "参考文献合集.txt"


def extract_text_pymupdf(pdf_path: Path) -> str:
    """优先使用 PyMuPDF 提取文本，效果好、速度快。"""
    import pymupdf  # 新版 PyMuPDF 的导入名（旧版为 fitz）

    parts = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts)


def extract_text_pypdf(pdf_path: Path) -> str:
    """备用方案：使用 pypdf 提取文本。"""
    from pypdf import PdfReader

    parts = []
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def extract_text(pdf_path: Path) -> str:
    """依次尝试 PyMuPDF 与 pypdf，返回提取到的文本。"""
    last_err = None
    for extractor in (extract_text_pymupdf, extract_text_pypdf):
        try:
            text = extractor(pdf_path)
            if text.strip():
                return text
        except Exception as err:  # noqa: BLE001 - 记录后尝试下一个提取器
            last_err = err
    if last_err is not None:
        print(f"  [警告] {pdf_path.name} 文本提取失败: {last_err}")
    return ""


def collect_pdfs(pdf_dir: Path) -> list[Path]:
    """递归收集目录下所有 PDF 文件，按相对路径排序，保证顺序稳定。"""
    pdfs = sorted(
        pdf_dir.rglob("*.pdf"),
        key=lambda p: str(p.relative_to(pdf_dir)).lower(),
    )
    return pdfs


def main() -> int:
    # 支持命令行覆盖默认参数
    pdf_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PDF_DIR
    output_txt = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_TXT

    if not pdf_dir.is_dir():
        print(f"[错误] 目录不存在: {pdf_dir}")
        return 1

    pdfs = collect_pdfs(pdf_dir)
    if not pdfs:
        print(f"[错误] 在 {pdf_dir} 下没有找到任何 PDF 文件。")
        return 1

    print(f"共找到 {len(pdfs)} 个 PDF 文件：")
    for p in pdfs:
        print(f"  - {p.relative_to(pdf_dir)}")

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    blocks = []
    for i, pdf_path in enumerate(pdfs, start=1):
        rel = pdf_path.relative_to(pdf_dir)
        print(f"[{i}/{len(pdfs)}] 处理: {rel}")
        text = extract_text(pdf_path)
        blocks.append(
            "\n".join(
                [
                    "=" * 80,
                    f"【第 {i} 篇】{rel}",
                    "=" * 80,
                    "",
                    text.strip(),
                    "",
                ]
            )
        )

    output_txt.write_text("\n".join(blocks), encoding="utf-8")
    print(f"\n完成！已生成: {output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
