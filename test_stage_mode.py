# -*- coding: utf-8 -*-
"""
测试 优化项4：答辩模式开关
1) preflight 在真实数据上全绿
2) preflight 能检测被破坏的数据（断链假设 / 未命中引用）
3) app.py 接线：开关存在、两个重算按钮受控、横幅与自检面板渲染
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

import preflight

failures = 0


def check(name, cond, detail=""):
    global failures
    mark = "✅" if cond else "❌"
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures += 1


# ── 1) 真实数据：全绿 ──────────────────────────────────────────
print("=== 1. 真实数据自检 ===")
r = preflight.run_preflight()
for cid, name, ok, detail in r["checks"]:
    check(f"{cid} {name}", ok, detail)
check("all_ok", r["all_ok"])

# ── 2) 破坏检测 ────────────────────────────────────────────────
print("=== 2. 数据破坏可被检测 ===")
tmp = tempfile.mkdtemp(prefix="rl_pf_")
try:
    # 健康 fixture：4 篇论文，2 个空白，quote 命中，假设回链正常
    pids = ["p1", "p2", "p3", "p4"]
    quote = "prolonged training for high-resolution images"
    seeds, mining, hyps = [], [], []
    for i, pid in enumerate(pids):
        txt = os.path.join(tmp, f"{pid}.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("padding words to reach minimum size. " * 100 + " " + quote)
        seeds.append({"paper_id": pid, "title": f"T{i}", "year": 2024,
                      "local_txt": txt, "core_contribution": "c"})
        mining.append({"paper_id": pid, "gaps": [
            {"gap_id": "G1", "gap_type": "方法局限", "gap_subtype": "计算开销大",
             "confidence": 0.8, "evidence": [{"quote": quote, "section": "S"}]},
            {"gap_id": "G2", "gap_type": "结论争议", "gap_subtype": "基线过时",
             "confidence": 0.7, "evidence": [{"quote": quote, "section": "S"}]},
        ], "_quote_check": {"hits": 2, "total": 2}})
        hyps.append({"cache_key": f"{pid}_budget30", "hypotheses": [
            {"hyp_id": "H1", "gap_id": "G1", "gate_passed": i % 2 == 0,
             "statement": "s", "total_score": 0.7},
        ]})

    def write_all(seed_rows, mining_rows, hyp_rows):
        for path, rows, key in (
            (os.path.join(tmp, "seed.jsonl"), seed_rows, "paper_id"),
            (os.path.join(tmp, "mining.jsonl"), mining_rows, "paper_id"),
            (os.path.join(tmp, "hyps.jsonl"), hyp_rows, "cache_key"),
        ):
            with open(path, "w", encoding="utf-8") as f:
                for o in rows:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")

    write_all(seeds, mining, hyps)
    saved = (preflight.SEED_PATH, preflight.MINING_PATH, preflight.HYP_PATH)
    preflight.SEED_PATH, preflight.MINING_PATH, preflight.HYP_PATH = (
        os.path.join(tmp, "seed.jsonl"),
        os.path.join(tmp, "mining.jsonl"),
        os.path.join(tmp, "hyps.jsonl"),
    )

    # 2a) 健康 fixture 应全绿
    rr = preflight.run_preflight()
    check("健康 fixture 全绿", rr["all_ok"],
          " | ".join(c[0] for c in rr["checks"] if not c[2]))

    # 2b) 断链假设 → P6 红
    broken_hyps = json.loads(json.dumps(hyps))
    broken_hyps[0]["hypotheses"][0]["gap_id"] = "G99"
    write_all(seeds, mining, broken_hyps)
    rr = preflight.run_preflight()
    by = {c[0]: c for c in rr["checks"]}
    check("P6 检测出断链假设", not by["P6"][2], by["P6"][3])
    check("P1-P5 不受影响仍绿", all(by[f"P{i}"][2] for i in range(1, 6)))

    # 2c) 假引用（未在原文命中）→ P4 红
    write_all(seeds, mining, hyps)  # 先恢复健康假设
    broken_mining = json.loads(json.dumps(mining))
    broken_mining[1]["gaps"][0]["evidence"][0]["quote"] = "never exists in any paper"
    write_all(seeds, broken_mining, hyps)
    rr = preflight.run_preflight()
    by = {c[0]: c for c in rr["checks"]}
    check("P4 检测出未命中引用", not by["P4"][2], by["P4"][3])

    # 2d) 缓存缺失 → P3/P5 红
    write_all(seeds, [], [])
    rr = preflight.run_preflight()
    by = {c[0]: c for c in rr["checks"]}
    check("P3 检测出挖掘缓存缺失", not by["P3"][2])
    check("P5 检测出假设缓存缺失", not by["P5"][2])

    preflight.SEED_PATH, preflight.MINING_PATH, preflight.HYP_PATH = saved
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ── 3) app.py 接线 ─────────────────────────────────────────────
print("=== 3. app.py 接线 ===")
src = open("app.py", encoding="utf-8").read()
import ast
ast.parse(src)  # 语法检查
check("开关定义存在", "stage_mode = st.toggle" in src)
check("挖掘按钮受答辩模式控制", "disabled=stage_mode" in src.split("开始文献挖掘")[1][:220])
check("假设按钮受答辩模式控制", "disabled=stage_mode" in src.split("生成假设")[1][:220])
check("自检面板渲染", "run_preflight" in src and "rl-pf-item" in src)
check("答辩横幅样式", "rl-stage-banner" in src or "答辩模式已开启" in src)

print("\n" + "=" * 60)
if failures:
    print(f"❌ {failures} 项失败")
    sys.exit(1)
print("✅ 答辩模式测试全部通过")
