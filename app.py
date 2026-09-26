# -*- coding: utf-8 -*-
import streamlit as st
import json, os
from attribution import attribute, load_cache
from decision import decide

st.set_page_config(page_title="退因 ReturnLens", page_icon="🔍", layout="wide")
st.title("🔍 退因 ReturnLens")
st.caption("3C 商家退货归因与逆向物流决策助手 · 端云混合架构 Demo")

if "results" not in st.session_state:
    st.session_state.results = {}

@st.cache_data
def load_tickets():
    path = os.path.join("data", "tickets_seed.jsonl")
    out = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line: out.append(json.loads(line))
    return out

tickets = load_tickets()
if not tickets:
    st.error("data/tickets_seed.jsonl 不存在，请先创建种子工单")
    st.stop()

ticket_id = st.sidebar.selectbox("选择工单", [t["id"] for t in tickets], index=0)
ticket = next(t for t in tickets if t["id"] == ticket_id)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**商品** {ticket['item']}")
st.sidebar.markdown(f"**售价** ¥{ticket['price']}  /  **进货价** ¥{ticket.get('cost_price', '—')}")
st.sidebar.markdown(f"**已激活** {'是 ✅' if ticket.get('activated') else '否'}")
st.sidebar.markdown(f"**物流** {ticket.get('logistics', '-')}")
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Demo 数据为模型辅助生成 + 人工注入噪声，非真实业务数据")

col1, col2 = st.columns([5, 7])

with col1:
    st.subheader("📥 工单原文")
    st.info(ticket["note"], icon="💬")
    for msg in ticket.get("chat", []):
        st.text(msg)
    force = st.checkbox("强制重新计算（忽略缓存）", key="force")
    if st.button("▶ 开始归因", type="primary"):
        with st.spinner("判责中…"):
            st.session_state.results[ticket_id] = attribute(ticket, use_cache=not force)
        st.rerun()

with col2:
    st.subheader("🧾 判定卡")
    out = st.session_state.results.get(ticket_id)
    if not out:
        st.warning("尚未归因，点击左侧按钮")
    elif "_error" in out:
        st.error("调用失败：" + out.get("_error", ""))
        if out.get("_raw"): st.code(out["_raw"])
    else:
        party = out.get("responsible_party", "-")
        st.markdown(f"### 责任方：{party}")
        pc = out.get("primary_cause") or {}
        conf = pc.get("confidence", 0)
        st.markdown(f"**主因** `{pc.get('level1','')}` / `{pc.get('level2','')}`　置信度 **{conf:.0%}**")
        if pc.get("evidence"):
            st.markdown("**证据引用**")
            for e in pc["evidence"]:
                st.markdown(f"> 「{e}」")
        rf = out.get("risk_flag")
        if rf: st.warning("⚠️ 风险标记：" + rf)
        if out.get("needs_more_evidence"):
            st.info("❓ 证据不足，需补充：" + str(out.get("ask_user", "—")))
        st.success("✅ " + str(out.get("actionable", "")))
        if out.get("_repaired"): st.caption("🔧 输出已自动修复：" + str(out["_repaired"]))
        with st.expander("查看原始 JSON"): st.json(out)

    # —— 决策引擎 ——
    st.divider()
    st.subheader("💰 处置方案净成本对比")
    if out and "_error" not in out:
        dec = decide(ticket, out)
        ranked = ([dec["recommend"]] if dec["recommend"] else []) + dec["alternatives"]
        for i, r in enumerate(ranked):
            tag = "🥇 推荐" if i == 0 else ("🥈 次优" if i == 1 else "🥉")
            bg = "#e6f4ea" if i == 0 else "#ffffff"
            st.markdown(
                f'<div style="background:{bg};border:1px solid #ddd;padding:10px 14px;border-radius:8px;margin:8px 0">'
                f"<b>{tag}</b> <b>{r['name']}</b>　净成本 <b>¥{r['cost']}</b>　风险 {r['risk']}</div>",
                unsafe_allow_html=True)
        if dec["sensitivity"]: st.info("💡 " + dec["sensitivity"])
        if dec["blocked"]:
            with st.expander(f"🚫 有 {len(dec['blocked'])} 个方案被归因规则拦截"):
                for b in dec["blocked"]:
                    st.markdown(f"- **{b['name']}**：{b['reason']}")