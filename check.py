# -*- coding: utf-8 -*-
"""
ResearchLens · 阶段1 验收脚本
检查项：API 联通 / 种子数据完整性 / 论文全文长度 / Limitations 段检出
"""

import os
import json
import sys
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("DASHSCOPE_API_KEY")
SEED_PATH = os.path.join("data", "lit_seed.jsonl")
PAPERS_DIR = os.path.join("data", "papers")

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
        passed += 1
    else:
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))
        failed += 1


# ── 1. API 联通 ──────────────────────────────────────────────
print("\n═══ 1. API 联通 ═══")
check("DASHSCOPE_API_KEY 已配置", bool(API_KEY),
      f"{API_KEY[:10]}..." if API_KEY else "未读取到 .env")

if API_KEY:
    try:
        from llm import call_qwen
        resp = call_qwen(
            [{"role": "user", "content": "请回答「收到」两个字"}],
            model="qwen-plus",
        )
        check("Qwen API 调用成功", resp is not None and "收到" in str(resp),
              f"回复: {resp}" if resp else "调用失败")
    except Exception as e:
        check("Qwen API 调用成功", False, str(e))

# ── 2. 种子数据完整性 ────────────────────────────────────────
print("\n═══ 2. 种子数据完整性 ═══")
papers = []
if os.path.exists(SEED_PATH):
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

check("lit_seed.jsonl 存在且有数据", len(papers) > 0, f"共 {len(papers)} 篇")
check("种子论文数量 ≥ 4", len(papers) >= 4)

required_fields = ["paper_id", "arxiv_id", "title", "year", "core_contribution",
                   "has_limitations_section", "limitations_extracted", "local_txt"]
for p in papers:
    pid = p.get("paper_id", "?")
    missing = [f for f in required_fields if f not in p]
    check(f"[{pid}] 字段完整", len(missing) == 0,
          f"缺失: {missing}" if missing else "OK")
    check(f"[{pid}] arXiv ID 格式合法",
          isinstance(p.get("arxiv_id"), str) and "." in p["arxiv_id"],
          p.get("arxiv_id", ""))

# ── 3. 论文全文文件 ──────────────────────────────────────────
print("\n═══ 3. 论文全文文件 ═══")
for p in papers:
    pid = p.get("paper_id", "?")
    txt_path = os.path.join(os.path.dirname(__file__) or ".", p.get("local_txt", ""))
    exists = os.path.exists(txt_path)
    check(f"[{pid}] 全文文件存在", exists, txt_path)

    if exists:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        char_count = len(content)
        check(f"[{pid}] 全文长度 ≥ 3000 字符", char_count >= 3000,
              f"{char_count} 字符")

        # Limitations 段检测（不区分大小写搜索关键词）
        has_lim = any(kw in content.lower() for kw in [
            "limitation", "future work", "limitation statement",
            "qualitative observation", "discussion"
        ])
        check(f"[{pid}] 含 Limitations 相关内容", has_lim)

# ── 4. 交叉引用闭合 ──────────────────────────────────────────
print("\n═══ 4. 交叉引用闭合 ═══")
all_ids = {p["paper_id"] for p in papers}
for p in papers:
    pid = p["paper_id"]
    refs_out = p.get("cross_refs_out", [])
    refs_in = p.get("cross_refs_in", [])
    for ref in refs_out + refs_in:
        check(f"[{pid}] 交叉引用 {ref} 存在于种子中", ref in all_ids)

# ── 5. taxonomy 枚举一致性 ──────────────────────────────────
print("\n═══ 5. taxonomy 枚举一致性 ═══")
try:
    from taxonomy import GAP_TYPE_LIST, HYPOTHESIS_KINDS, RESOURCE_TAGS, VERIFIABLE_VIA
    check("GAP_TYPE_LIST 非空", len(GAP_TYPE_LIST) >= 4, f"{len(GAP_TYPE_LIST)} 类")
    check("HYPOTHESIS_KINDS 非空", len(HYPOTHESIS_KINDS) >= 4, f"{len(HYPOTHESIS_KINDS)} 种")
    check("RESOURCE_TAGS 非空", len(RESOURCE_TAGS) >= 4, f"{len(RESOURCE_TAGS)} 种")
    check("VERIFIABLE_VIA 非空", len(VERIFIABLE_VIA) >= 4, f"{len(VERIFIABLE_VIA)} 种")
except ImportError as e:
    check("taxonomy.py 可导入", False, str(e))

# ── 汇总 ──────────────────────────────────────────────────────
print(f"\n{'═' * 50}")
print(f"  通过: {passed}  失败: {failed}  总计: {passed + failed}")
if failed:
    print("  ⚠️ 有失败项，请检查后重试")
    sys.exit(1)
else:
    print("  🎉 阶段1 验收全部通过！")
