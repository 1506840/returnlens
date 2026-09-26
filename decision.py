# -*- coding: utf-8 -*-
"""
决策引擎 v2：归因门控 + 净成本模型 + 敏感性
"""

DEFAULT_PARAMS = {
    "return_shipping": 12, "outbound_shipping": 10,
    "inspection": 8, "restock": 5,
    "residual_unopened": 0.95, "residual_opened": 0.75, "residual_activated": 0.55,
    "cx_low": 5, "cx_high": 25,
    "self_help_service": 15,   # 引导自助一轮的客服沟通成本
}

SCHEMES = [
    {"key": "return_refund", "name": "退货退款",       "risk": "低"},
    {"key": "refund_only",   "name": "仅退款不退货",   "risk": "高"},
    {"key": "replace_first", "name": "换货（先发新）", "risk": "中高"},
    {"key": "resend_part",   "name": "补发配件/耗材",  "risk": "低"},
    {"key": "partial_keep",  "name": "部分退款留用",   "risk": "低"},
    {"key": "self_help",     "name": "引导自助解决",   "risk": "中"},
]


def _residual_loss(cost, rate):
    return cost * (1 - rate)


def _cost(ticket):
    return ticket.get("cost_price") or ticket["price"] * 0.65


def _pc(attr):
    return attr.get("primary_cause") or {}


def _discount(l2):
    """部分退款留用的折价比例，按二级标签细分"""
    if l2 in ("与描述不符",):
        return 0.15     # 色差/描述不符，折价小
    if l2 in ("主观不喜欢", "买错型号规格"):
        return 0.30
    return 0.20


def _fail_rate(l1, l2, party):
    """引导自助的失败率：硬件/物流问题自助大概率解决不了"""
    if l1 == "用户原因" and l2 == "不会用或配置不当":
        return 0.30
    if l2 in ("已激活后悔", "不会用或配置不当"):
        return 0.45
    if party == "无法判定":
        return 0.70
    if l1 == "商品问题":
        return 0.80     # 硬件坏了，自助基本无效
    return 0.60


def _eligible(key, ticket, attr):
    """归因门控：返回 (业务上是否可行, 不可行原因)"""
    pc = _pc(attr)
    l1, l2 = pc.get("level1"), pc.get("level2")
    party = attr.get("responsible_party")
    act = ticket.get("activated", False)
    need = attr.get("needs_more_evidence", False)

    if key == "self_help":
        ok = (l1 == "用户原因") or (act and l2 == "功能故障") or (need and party == "无法判定")
        return ok, "商品硬件/物流类问题不应引导用户自助"
    if key == "resend_part":
        ok = (l1 == "商品问题" and l2 == "配件错漏发")
        return ok, "仅『配件错漏发』适合补发"
    if key == "partial_keep":
        keepable = ("与描述不符", "主观不喜欢", "买错型号规格")   # 商品本身没坏才允许折价留用
        ok = (l2 in keepable) and not act
        return ok, "仅『未损坏、主观或描述偏差』类可折价留用；硬件故障/破损/配件缺失不适用"
    if key == "refund_only":
        return False, "3C 商品仅退款不退货成本最高，默认关闭"
    if key == "replace_first":
        ok = (l1 == "商品问题" and not need)
        return ok, "证据不足时不先发改新（防止旧件不退）"
    if key == "return_refund":
        return True, ""   # 兜底方案，始终可行
    return False, ""


def _raw_cost(key, ticket, attr, p, fallback_return):
    """纯成本核算（不看门控）"""
    cost, price = _cost(ticket), ticket["price"]
    act = ticket.get("activated", False)
    pc = _pc(attr)
    res = p["residual_activated"] if act else p["residual_opened"]
    lines, total = {}, 0.0

    if key == "return_refund":
        total = p["return_shipping"] + p["inspection"] + p["restock"] + _residual_loss(cost, res) + p["cx_low"]
        lines = {"逆向运费": p["return_shipping"], "质检+上架": p["inspection"] + p["restock"],
                 f"残值损失(残值率{res})": round(_residual_loss(cost, res), 1), "体验折损": p["cx_low"]}
    elif key == "refund_only":
        total = cost + p["cx_high"]
        lines = {"商品损失(全额)": cost, "体验折损": p["cx_high"]}
    elif key == "replace_first":
        total = p["outbound_shipping"] * 2 + p["inspection"] + p["restock"] + _residual_loss(cost, res) + p["cx_low"]
        lines = {"双向运费": p["outbound_shipping"] * 2, "质检+上架": p["inspection"] + p["restock"],
                 "旧件残值损失": round(_residual_loss(cost, res), 1), "体验折损": p["cx_low"],
                 "⚠️旧件未收回": "需人工确认"}
    elif key == "resend_part":
        part = price * 0.08
        total = part + p["outbound_shipping"] + p["cx_low"]
        lines = {"配件成本(估8%)": round(part, 1), "运费": p["outbound_shipping"], "体验折损": p["cx_low"]}
    elif key == "partial_keep":
        disc = _discount(pc.get("level2"))
        refund = price * disc
        total = refund + p["cx_low"]
        lines = {f"折价退款({int(disc*100)}%)": round(refund, 1), "体验折损": p["cx_low"]}
    elif key == "self_help":
        fr = _fail_rate(pc.get("level1"), pc.get("level2"), attr.get("responsible_party"))
        total = p["self_help_service"] + fr * fallback_return
        lines = {"自助沟通成本": p["self_help_service"],
                 f"失败率{fr:.0%}×兜底退货成本": round(fr * fallback_return, 1)}
    return round(total, 1), lines


def decide(ticket, attr, params=None):
    p = params or DEFAULT_PARAMS
    # 先算兜底退货成本，供 self_help 期望成本使用
    fb, _ = _raw_cost("return_refund", ticket, attr, p, 0)

    rows, blocked = [], []
    for s in SCHEMES:
        ok, reason = _eligible(s["key"], ticket, attr)
        c, lines = _raw_cost(s["key"], ticket, attr, p, fb)
        row = {"key": s["key"], "name": s["name"], "cost": c, "details": lines,
               "risk": s["risk"], "eligible": ok, "reason": reason}
        if ok:
            rows.append(row)
        else:
            blocked.append({"name": s["name"], "reason": reason})
    rows.sort(key=lambda r: r["cost"])
    best = rows[0] if rows else None
    second = rows[1] if len(rows) > 1 else None

    # 敏感性：残值率临界点
    sensitivity = ""
    if ticket.get("activated"):
        sensitivity = (f"敏感性：该商品已激活，残值率按 {p['residual_activated']} 估算。"
                       f"若实际残值率低于约 0.45，逆向回收价值进一步走低，建议优先低成本动作而非退货。")
    elif best and second:
        sensitivity = (f"当前未激活，残值率按 {p['residual_opened']} 估算；"
                       f"推荐「{best['name']}」相对次优「{second['name']}」省约 ¥{second['cost']-best['cost']:.0f}。")

    return {
        "recommend": best,
        "alternatives": rows[1:3],
        "all": rows,
        "blocked": blocked,            # 被归因门控拦截的方案 + 原因
        "sensitivity": sensitivity,
        "params_used": p,
    }


if __name__ == "__main__":
    import json, os
    from attribution import attribute, load_cache

    path = os.path.join("data", "tickets_seed.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            ticket = json.loads(line)
            attr = load_cache().get(ticket["id"]) or attribute(ticket)
            out = decide(ticket, attr)
            pc = attr.get("primary_cause") or {}
            print(f"\n===== {ticket['id']} | {ticket['item']} | "
                  f"主因:{pc.get('level1')}/{pc.get('level2')} | 责任方:{attr.get('responsible_party')} =====")
            if out["recommend"]:
                print(f"🥇 推荐 {out['recommend']['name']}  净成本 ¥{out['recommend']['cost']}")
            for a in out["alternatives"]:
                print(f"🥈 备选 {a['name']}  净成本 ¥{a['cost']}")
            if out["sensitivity"]:
                print(f"💡 {out['sensitivity']}")
            for b in out["blocked"]:
                print(f"🚫 拦截 {b['name']} —— {b['reason']}")