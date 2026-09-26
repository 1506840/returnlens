# -*- coding: utf-8 -*-
"""周复盘：把归因结果聚合成 Top 损失源 / 责任方分布 / 风险清单"""

import sys
import json
import os
import subprocess
from collections import defaultdict

RESULTS_PATH = os.path.join("data", "results.jsonl")
TICKETS_PATH = os.path.join("data", "tickets_seed.jsonl")
PARAMS = {"residual_opened": 0.75, "residual_activated": 0.55}


def _cost(ticket):
    return ticket.get("cost_price") or ticket["price"] * 0.65


def _estimate_loss(ticket, attr, p):
    price, cost = ticket["price"], _cost(ticket)
    res = p["residual_activated"] if ticket.get("activated") else p["residual_opened"]
    pc = attr.get("primary_cause") or {}
    l1, l2 = pc.get("level1"), pc.get("level2")
    need = attr.get("needs_more_evidence", False)
    if need:
        return 15.0
    if l2 in ("功能故障", "批次不良", "破损"):
        return cost * (1 - res) + 25
    if l2 in ("与描述不符", "主观不喜欢", "买错型号规格"):
        return price * 0.18 + 12
    if l1 == "用户原因":
        return price * 0.10 + 12
    return price * 0.15


def weekly_review(auto_regen=True):
    # ① 种子工单
    if not os.path.exists(TICKETS_PATH):
        print(f"❌ 缺少 {TICKETS_PATH}")
        return None
    tickets = {}
    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                tickets[t["id"]] = t
    print(f"✅ 读到 {len(tickets)} 条种子工单: {list(tickets.keys())}")

    # ② 归因结果
    if not os.path.exists(RESULTS_PATH):
        print(f"❌ 缺少 {RESULTS_PATH}")
        if auto_regen and os.path.exists("attribution.py"):
            print("🔄 自动补跑 attribution.py ...")
            subprocess.run([sys.executable, "attribution.py"], check=False)
        if not os.path.exists(RESULTS_PATH):
            print("❌ 补跑后仍无结果文件")
        return None
    print(f"✅ 找到 {RESULTS_PATH}")

    results = []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"⚠️ 第{i}行 JSON 解析失败: {e}")
                continue
            tid = obj.get("id")
            if tid is None:
                print(f"⚠️ 第{i}行缺少 id 字段（attribution 未注入）→ 跳过")
                continue
            if "_error" in obj:
                print(f"⚠️ 跳过 {tid}: {obj['_error']}")
                continue
            if tid not in tickets:
                print(f"⚠️ {tid} 在种子工单中找不到 → 跳过")
                continue
            results.append({"ticket": tickets[tid], "attr": obj})
    print(f"✅ 有效匹配 {len(results)} 条工单\n")

    if not results:
        return None

    n = len(results)
    total_loss = sum(_estimate_loss(r["ticket"], r["attr"], PARAMS) for r in results)

    cause_agg = defaultdict(lambda: {"count": 0, "loss": 0.0})
    party_agg = defaultdict(int)
    risks = []
    for r in results:
        pc = r["attr"].get("primary_cause") or {}
        key = f"{pc.get('level1', '未知')}/{pc.get('level2', '其他')}"
        loss = _estimate_loss(r["ticket"], r["attr"], PARAMS)
        cause_agg[key]["count"] += 1
        cause_agg[key]["loss"] += loss
        party_agg[r["attr"].get("responsible_party", "未知")] += 1
        if r["attr"].get("risk_flag"):
            risks.append((r["ticket"]["id"], r["ticket"]["item"], r["attr"]["risk_flag"]))

    top_causes = sorted(cause_agg.items(), key=lambda x: -x[1]["loss"])
    name, info = top_causes[0]
    pct = info["loss"] / total_loss * 100 if total_loss else 0
    conclusion = (f"本周 {n} 单中，「{name}」是最大损失源，"
                  f"约占 {pct:.0f}%（¥{info['loss']:.0f}），建议优先排查该类目供应商。")

    return {
        "n": n,
        "total_loss": round(total_loss, 1),
        "top_causes": [(k, int(v["count"]), round(v["loss"], 1)) for k, v in top_causes],
        "party_dist": dict(party_agg),
        "risks": risks,
        "conclusion": conclusion,
    }


if __name__ == "__main__":
    out = weekly_review()
    if not out:
        print("\n❌ 复盘生成失败，请检查上方日志")
        sys.exit(1)
    print(f"本周工单数: {out['n']}  估算总损失: ¥{out['total_loss']:,.0f}\n")
    print("【Top 损失源】")
    for k, c, v in out["top_causes"]:
        bar = "█" * max(1, int(v / out["total_loss"] * 30))
        print(f"  {k:24} {c:>3}单  ¥{v:>8,.0f}  {bar}")
    print(f"\n【责任方分布】 {out['party_dist']}")
    print(f"\n【风险单】共 {len(out['risks'])} 条")
    for tid, item, rf in out["risks"]:
        print(f"  ⚠️ {tid} {item}：{rf}")
    print(f"\n💡 {out['conclusion']}")