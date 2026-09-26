# -*- coding: utf-8 -*-
"""
周复盘模块：把归因结果聚合成经营视角的周报
- 纯本地计算，不调用大模型，可反复运行、结果稳定
- 结论句按「一级原因」分支给建议（修复：用户原因不再建议排查供应商）
"""

import os
import json

TICKETS_PATH = os.path.join("data", "tickets_seed.jsonl")
RESULTS_PATH = os.path.join("data", "results.jsonl")

# —— 成本参数（经验值，生产环境应用历史退货数据拟合）——
REVERSE_LOGISTICS = 12.0   # 逆向物流
QC_LABOR = 15.0            # 质检 + 人工处理
SALVAGE_RATE = {"activated": 0.55, "not_activated": 0.75}
# 商家实际承担比例（按责任方）：供应商/物流商责任可追偿，仅计垫付损耗
PARTY_BEAR = {
    "供应商": 0.15,
    "物流商": 0.15,
    "仓配": 0.20,
    "运营": 0.60,
    "商家自身": 1.00,
    "用户": 0.35,
    "平台规则": 0.50,
    "无法判定": 0.50,
}


def load_tickets():
    out = []
    if os.path.exists(TICKETS_PATH):
        with open(TICKETS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def load_results():
    out = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    out[o.get("id")] = o
                except Exception:
                    pass
    return out


def estimate_loss(ticket, result):
    """估算该单给商家带来的净损失"""
    price = float(ticket.get("price", 0) or 0)
    rate = SALVAGE_RATE["activated"] if ticket.get("activated") else SALVAGE_RATE["not_activated"]
    gross = price * (1 - rate) + REVERSE_LOGISTICS + QC_LABOR
    party = result.get("responsible_party", "无法判定")
    return gross * PARTY_BEAR.get(party, 0.5)


def build_conclusion(top_causes, party_dist, n, total_loss):
    """按一级原因给出对应建议，避免『用户原因却建议排查供应商』的逻辑矛盾"""
    if not top_causes:
        return f"本周 {n} 单，暂无可归因数据。"
    cause, cnt, loss = top_causes[0]
    share = (loss / total_loss * 100) if total_loss else 0
    l1 = cause.split("/", 1)[0]
    advice = {
        "商品问题": "建议优先排查该类目供应商来料与质检，并核对详情页描述是否充分",
        "物流问题": "建议优先排查承运商时效与包装规范，必要时调整仓配方案",
        "用户原因": "建议优先优化详情页与售前引导（规格、颜色、兼容性说明），降低主观退货率",
        "商家服务": "建议优先复盘客服话术与承诺兑现流程",
        "规则性": "建议优先前置展示退换货规则，减少规则争议",
    }.get(l1, "建议优先排查该类目主要损失来源")
    return (f"本周 {n} 单中，「{cause}」是最大损失源，"
            f"约占 {share:.0f}%（¥{loss:,.0f}），{advice}。")


def weekly_review(auto_regen=True, verbose=True):
    tickets = load_tickets()
    if not tickets:
        if verbose:
            print("❌ 未读到种子工单", file=os.sys.stderr if hasattr(os, "sys") else None)
        return None

    results = load_results()

    # results 缺失时可选自动补跑（界面里传 auto_regen=False，避免演示时意外重算）
    if not results and auto_regen:
        try:
            from attribution import attribute
            for t in tickets:
                attribute(t, use_cache=False)
            results = load_results()
        except Exception as e:
            if verbose:
                print(f"⚠️ 自动归因失败：{e}")

    if not results:
        if verbose:
            print("❌ 无归因结果，请先运行 attribution.py")
        return None

    items = []
    for t in tickets:
        r = results.get(t["id"])
        if not r or "_error" in r:
            continue
        pc = r.get("primary_cause") or {}
        cause = f"{pc.get('level1', '未知')}/{pc.get('level2', '未知')}"
        items.append({
            "id": t["id"],
            "item": t.get("item", ""),
            "cause": cause,
            "party": r.get("responsible_party", "无法判定"),
            "loss": estimate_loss(t, r),
            "risk": r.get("risk_flag"),
            "needs_more_evidence": bool(r.get("needs_more_evidence")),
        })

    if not items:
        if verbose:
            print("❌ 有效匹配 0 条")
        return None

    n = len(items)
    total_loss = sum(x["loss"] for x in items)

    # Top 损失源（按损失额排序）
    agg = {}
    for x in items:
        a = agg.setdefault(x["cause"], [0, 0.0])
        a[0] += 1
        a[1] += x["loss"]
    top_causes = sorted(
        [(k, v[0], v[1]) for k, v in agg.items()],
        key=lambda z: z[2], reverse=True,
    )

    party_dist = {}
    for x in items:
        party_dist[x["party"]] = party_dist.get(x["party"], 0) + 1

    risks = [(x["id"], x["item"], x["risk"]) for x in items if x["risk"]]

    return {
        "n": n,
        "total_loss": total_loss,
        "top_causes": top_causes,
        "party_dist": party_dist,
        "risks": risks,
        "conclusion": build_conclusion(top_causes, party_dist, n, total_loss),
    }


if __name__ == "__main__":
    tickets = load_tickets()
    print(f"✅ 读到 {len(tickets)} 条种子工单: {[t['id'] for t in tickets]}")
    print(f"{'✅ 找到' if os.path.exists(RESULTS_PATH) else '❌ 缺少'} {RESULTS_PATH}")

    rev = weekly_review(auto_regen=False)
    if not rev:
        raise SystemExit(1)

    print(f"✅ 有效匹配 {rev['n']} 条工单\n")
    print(f"本周工单数: {rev['n']}  估算总损失: ¥{rev['total_loss']:,.0f}\n")

    print("【Top 损失源】")
    max_loss = max(c[2] for c in rev["top_causes"]) or 1
    for cause, cnt, loss in rev["top_causes"]:
        bar = "█" * max(1, round(loss / max_loss * 15))
        print(f"  {cause:<24}{cnt}单  ¥{loss:>8,.0f}  {bar}")

    print(f"\n【责任方分布】 {rev['party_dist']}\n")

    print(f"【风险单】共 {len(rev['risks'])} 条")
    for tid, item, rf in rev["risks"]:
        print(f"  ⚠️ {tid} {item}：{rf}")

    print(f"\n💡 {rev['conclusion']}")