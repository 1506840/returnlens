# -*- coding: utf-8 -*-
"""
测试 优化项3：假设→证据反向回链
1) index_gaps / gap_card_html 单元测试（AST 提取，不启动 streamlit）
2) 回链解析边界情况（target_gap_ids 优先 / gap_id 兜底）
3) 可追溯性：data/hypotheses.jsonl 每条假设的 gap 必须能在来源论文的
   挖掘结果中解析到带逐字证据的空白卡
4) 回链闭环：空白卡的引用必须仍能在论文全文中逐字命中
"""
import ast
import json
import re
import sys

sys.path.insert(0, ".")

# ── 从 app.py 提取共享函数（不执行 streamlit 主体）──────────────────
src = open("app.py", encoding="utf-8").read()
tree = ast.parse(src)
want = ("index_gaps", "gap_card_html", "conf_badge", "html_escape")
funcs = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        funcs[node.name] = ast.get_source_segment(src, node)
missing = [f for f in want if f not in funcs]
assert not missing, f"app.py 缺少函数: {missing}"

# 接线检查：三个回链函数都应在 app.py 中定义并被调用
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
called = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        f = node.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if name:
            called.add(name)
for fn in ("index_gaps", "gap_card_html", "jump_button"):
    assert fn in defined, f"{fn} 未在 app.py 中定义"
for fn in ("index_gaps", "gap_card_html", "jump_button"):
    assert fn in called, f"{fn} 未在 app.py 渲染流程中被调用"

THEME = {
    "primary": "#6C5CE7", "primary_dark": "#4834D4", "accent": "#00B894",
    "warn": "#FDCB6E", "danger": "#E17055", "bg": "#F5F6FA", "card": "#FFFFFF",
    "text": "#2D3436", "muted": "#636E72",
}
ns = {"THEME": THEME}
for fn in ("conf_badge", "html_escape", "index_gaps", "gap_card_html"):
    exec(funcs[fn], ns)
index_gaps, gap_card_html = ns["index_gaps"], ns["gap_card_html"]

failures = 0


def check(name, cond, detail=""):
    global failures
    if cond:
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    else:
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))
        failures += 1


# ── 1) 单元测试 ────────────────────────────────────────────────
print("=== 1. index_gaps 与 gap_card_html 单元 ===")
gaps_doc = {
    "gaps": [
        {"gap_id": "G1", "gap_type": "方法局限", "gap_subtype": "计算开销大",
         "confidence": 0.8, "evidence": [{"quote": "abc", "section": "Limitations"}]},
        {"gap_id": "G2", "gap_type": "结论争议", "gap_subtype": "基线过时",
         "confidence": 0.4, "evidence": []},
    ]
}
idx = index_gaps(gaps_doc)
check("索引覆盖全部 gap_id", set(idx) == {"G1", "G2"}, str(sorted(idx)))
check("None/空文档容错", index_gaps(None) == {} and index_gaps({}) == {})

html = gap_card_html(gaps_doc["gaps"][0])
check("含 ID 与类型", "G1" in html and "方法局限" in html)
check("高置信徽章（>0.6）", "高置信" in html and "80%" in html)
check("含逐字引用", "abc" in html and "Limitations" in html)
html2 = gap_card_html(gaps_doc["gaps"][1])
check("低置信徽章（≤0.4）", "低置信" in html2 and "40%" in html2)

# ── 2) 回链解析边界情况（与 app.jump_button 同逻辑）──────────────
print("=== 2. 回链 ID 解析边界 ===")


def resolve_gap_id(h):
    gid = h.get("gap_id") or ""
    tg = h.get("target_gap_ids")
    if isinstance(tg, list) and tg:
        gid = str(tg[0])
    elif tg and not gid:
        gid = str(tg)
    return gid


check("target_gap_ids 列表优先", resolve_gap_id({"gap_id": "G1", "target_gap_ids": ["G2", "G3"]}) == "G2")
check("无 target 回退 gap_id", resolve_gap_id({"gap_id": "G5"}) == "G5")
check("仅 target 字符串", resolve_gap_id({"target_gap_ids": "G7"}) == "G7")
check("都为空返回空串", resolve_gap_id({}) == "")

# ── 3) 端到端可追溯性（真实数据）────────────────────────────────
print("=== 3. 真实假设数据可追溯性 ===")
with open("data/mining_results.jsonl", encoding="utf-8") as f:
    mining = {json.loads(l)["paper_id"]: json.loads(l) for l in f if l.strip()}
with open("data/hypotheses.jsonl", encoding="utf-8") as f:
    cache = {}
    for l in f:
        if l.strip():
            o = json.loads(l)
            cache[o["cache_key"]] = o["hypotheses"]

total = resolved = 0
for key, hyps in cache.items():
    pid = re.match(r"(.+)_budget", key).group(1)
    doc = mining.get(pid)
    check(f"{key} 对应挖掘结果存在", doc is not None)
    if not doc:
        continue
    gidx = index_gaps(doc)
    for h in hyps:
        total += 1
        gid = resolve_gap_id(h)
        g = gidx.get(gid)
        if g and isinstance(g.get("evidence"), list) and len(g["evidence"]) > 0:
            resolved += 1
        else:
            print(f"     ⚠️ {h.get('hyp_id')} gap={gid!r} 无法解析（{pid} 有 {list(gidx)}）")

check("全部假设可回链到带证据的空白卡", resolved == total and total > 0, f"{resolved}/{total}")

# 空白卡字段完整性
field_ok = True
for pid, doc in mining.items():
    for g in doc.get("gaps", []):
        for field in ("gap_id", "gap_type", "gap_subtype", "confidence", "evidence"):
            if field not in g:
                field_ok = False
        for ev in g.get("evidence", []):
            if not (isinstance(ev, dict) and "quote" in ev and "section" in ev):
                field_ok = False
check("14 个空白卡字段齐全", field_ok)

# ── 4) 回链闭环：空白引用命中论文原文 ───────────────────────────
print("=== 4. 回链闭环到论文原文 ===")
with open("data/lit_seed.jsonl", encoding="utf-8") as f:
    seeds = {json.loads(l)["paper_id"]: json.loads(l) for l in f if l.strip()}

def norm(s):
    s = str(s).replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(s.lower().split())

for pid, doc in mining.items():
    paper = seeds.get(pid)
    if not paper:
        check(f"{pid} 种子存在", False)
        continue
    qc = doc.get("_quote_check") or {}
    # 独立复核：逐条 quote 在全文命中（引号归一化 + 空白折叠）
    full_norm = norm(open(paper["local_txt"], encoding="utf-8").read())
    hits = sum(
        1
        for g in doc.get("gaps", [])
        for ev in (g.get("evidence") or [])
        if isinstance(ev, dict) and norm(ev.get("quote", "")) in full_norm
    )
    total_q = sum(len(g.get("evidence") or []) for g in doc.get("gaps", []))
    check(f"{pid} 空白引用逐字命中原文", hits == total_q and hits == qc.get("hits", -1),
          f"{hits}/{total_q}")

print("\n" + "=" * 60)
if failures:
    print(f"❌ {failures} 项失败")
    sys.exit(1)
print(f"✅ 优化项3 全部通过（{resolved}/{total} 条假设可回链到证据，证据闭环命中原文）")
