# -*- coding: utf-8 -*-
"""
ResearchLens · 阶段1 验收脚本
检查项：
  1) data/lit_seed.jsonl 恰好 4 篇、字段完整、arXiv ID 合法
  2) data/papers/*.txt 每篇 >= 3000 字符，且含 Limitations / Future Work 段
  3) taxonomy.py 枚举封闭性（lit_seed 引用的分类学字段若存在需在枚举内）
  4) DASHSCOPE_API_KEY 是否存在（API 联通用 check_api 单独跑）
用法：python check_stage1.py
"""

import os
import re
import json
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
PAPERS_DIR = os.path.join(DATA, "papers")
LIT_SEED = os.path.join(DATA, "lit_seed.jsonl")

MIN_CHARS = 3000
LIMITATION_PAT = re.compile(
    r"(Limitations?|Future Work|Future Directions|附录\s*[A-Z]\s*Limitations)",
    re.IGNORECASE,
)

REQUIRED_FIELDS = [
    "paper_id", "arxiv_id", "title", "year", "role",
    "core_contribution", "has_limitations_section",
    "limitations_location", "limitations_extracted",
    "cross_refs_out", "cross_refs_in", "local_txt",
]


def load_seed():
    rows = []
    with open(LIT_SEED, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_seed(rows):
    errs = []
    if len(rows) != 4:
        errs.append(f"lit_seed 应为 4 篇，实际 {len(rows)}")
    ids = set()
    for r in rows:
        pid = r.get("paper_id", "?")
        for fld in REQUIRED_FIELDS:
            if fld not in r:
                errs.append(f"[{pid}] 缺字段 {fld}")
        if not re.fullmatch(r"\d{4}\.\d{4,5}", str(r.get("arxiv_id", ""))):
            errs.append(f"[{pid}] arxiv_id 非法: {r.get('arxiv_id')}")
        if not r.get("has_limitations_section"):
            errs.append(f"[{pid}] 未标注含 Limitations 段")
        if len(r.get("limitations_extracted") or []) < 3:
            errs.append(f"[{pid}] limitations_extracted 少于 3 条")
        if pid in ids:
            errs.append(f"paper_id 重复: {pid}")
        ids.add(pid)
    # 交叉引用闭合：引用出去的 id 必须在本集合内
    for r in rows:
        for ref in (r.get("cross_refs_out") or []) + (r.get("cross_refs_in") or []):
            if ref not in ids:
                errs.append(f"[{r['paper_id']}] 交叉引用指向未知论文 {ref}")
    return errs, ids


def check_papers(rows):
    errs = []
    for r in rows:
        path = os.path.join(BASE, r["local_txt"].replace("/", os.sep))
        if not os.path.exists(path):
            errs.append(f"[{r['paper_id']}] 全文文件缺失: {r['local_txt']}")
            continue
        txt = open(path, encoding="utf-8").read()
        n = len(txt)
        if n < MIN_CHARS:
            errs.append(f"[{r['paper_id']}] 全文仅 {n} 字符 (<{MIN_CHARS})")
        if not LIMITATION_PAT.search(txt):
            errs.append(f"[{r['paper_id']}] 全文未检出 Limitations/Future Work 段")
        print(f"  {r['paper_id']:<10} {n:>7} chars  limitations={'YES' if LIMITATION_PAT.search(txt) else 'NO '}")
    return errs


def check_taxonomy_closure():
    errs = []
    try:
        import taxonomy as T
    except Exception as e:
        return [f"无法导入 taxonomy: {e}"]
    if sum(T.SCORE_WEIGHTS.values()) != 1:
        errs.append(f"SCORE_WEIGHTS 之和 {sum(T.SCORE_WEIGHTS.values())} != 1")
    if set(T.SCORE_WEIGHTS) != set(T.SCORE_DIMENSIONS):
        errs.append("SCORE_WEIGHTS 与 SCORE_DIMENSIONS 维度不一致")
    for k, v in T.GAP_TYPES.items():
        if not v:
            errs.append(f"GAP_TYPES[{k}] 为空")
    if len(set(T.HYPOTHESIS_KINDS)) != len(T.HYPOTHESIS_KINDS):
        errs.append("HYPOTHESIS_KINDS 有重复")
    return errs


def main():
    print("== ResearchLens 阶段1 验收 ==")
    all_errs = []

    if not os.path.exists(LIT_SEED):
        print("FAIL: data/lit_seed.jsonl 不存在"); sys.exit(1)

    rows = load_seed()
    e, _ = check_seed(rows); all_errs += e
    print(f"[1] lit_seed.jsonl 4 篇校验: {'OK' if not e else 'FAIL'}")

    e = check_papers(rows); all_errs += e
    print(f"[2] 全文长度>= {MIN_CHARS} 且含 Limitations 段: {'OK' if not e else 'FAIL'}")

    e = check_taxonomy_closure(); all_errs += e
    print(f"[3] taxonomy 枚举一致性: {'OK' if not e else 'FAIL'}")

    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not key:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    print(f"[4] DASHSCOPE_API_KEY: {'已配置' if key else '未配置（需 .env）'}")

    if all_errs:
        print("\n--- 待修问题 ---")
        for x in all_errs:
            print(" -", x)
        sys.exit(1)
    print("\n阶段1 静态验收全部通过。API 实调请运行 python check_api.py")


if __name__ == "__main__":
    main()
