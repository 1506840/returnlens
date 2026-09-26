# -*- coding: utf-8 -*-
"""
退因 ReturnLens · Streamlit 界面
注意：本文件不使用 st.table / st.dataframe / st.bar_chart / st.line_chart，
      因为它们底层依赖 pandas，当前环境 pandas DLL 被系统策略拦截。
"""

import os
import json
import streamlit as st

from attribution import attribute
from decision import decide
from review import weekly_review

st.set_page_config(page_title="退因 ReturnLens", page_icon="🔍", layout="wide")
st.title("🔍 退因 ReturnLens")
st.caption("3C 商家退货归因与逆向物流决策助手 · 端云混合架构 Demo")

# ---------- 会话状态 ----------
if "results" not in st.session_state:
    st.session_state.results = {}
if "show_review" not in st.session_state:
    st.session_state.show_review = False


# ---------- 数据加载 ----------
@st.cache_data
def load_tickets():
    path = os.path.join("data", "tickets_seed.jsonl")
    out = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


tickets = load_tickets()
if not tickets:
    st.error("data/tickets_seed.jsonl 不存在，请先创建种子工单")
    st.stop()

ticket_id = st.sidebar.selectbox("选择工单", [t["id"] for t in tickets], index=0)
ticket = next(t for t in tickets if t["id"] == ticket_id)

# ---------- 侧边栏 ----------
st.sidebar.markdown("---")
st.sidebar.markdown(f"**商品** {ticket['item']}")
st.sidebar.markdown(f"**售价** ¥{ticket['price']}  /  **进货价** ¥{ticket.get('cost_price', '—')}")
st.sidebar.markdown(f"**已激活** {'是 ✅' if ticket.get('activated') else '否'}")
st.sidebar.markdown(f"**物流** {ticket.get('logistics', '-')}")
st.sidebar.markdown("---")
if st.sidebar.button("📊 生成周复盘", use_container_width=True):
    st.session_state.show_review = True
    st.rerun()
st.sidebar.caption("⚠️ Demo 数据为模型辅助生成 + 人工注入噪声，非真实业务数据")

# ---------- 主区：左工单 / 右判定 ----------
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
        if out.get("_raw"):
            st.code(out["_raw"])
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
        if rf:
            st.warning("⚠️ 风险标记：" + rf)
        if out.get("needs_more_evidence"):
            st.info("❓ 证据不足，需补充：" + str(out.get("ask_user", "—")))
        st.success("✅ " + str(out.get("actionable", "")))
        if out.get("_repaired"):
            st.caption("🔧 输出已自动修复：" + str(out["_repaired"]))
        with st.expander("查看原始 JSON"):
            st.json(out)

    # ---------- 决策引擎 ----------
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
                unsafe_allow_html=True,
            )
        if dec["sensitivity"]:
            st.info("💡 " + dec["sensitivity"])
        if dec["blocked"]:
            with st.expander(f"🚫 有 {len(dec['blocked'])} 个方案被归因规则拦截"):
                for b in dec["blocked"]:
                    st.markdown(f"- **{b['name']}**：{b['reason']}")

    if st.button("💾 导出本轮结果", key="export"):
        os.makedirs("data", exist_ok=True)
        with open(os.path.join("data", "results.jsonl"), "w", encoding="utf-8") as f:
            for tid, res in st.session_state.results.items():
                f.write(json.dumps({"id": tid, **res}, ensure_ascii=False) + "\n")
        st.success("已保存到 data/results.jsonl")

# ---------- 周复盘（HTML 实现，不依赖 pandas） ----------
if st.session_state.show_review:
    st.divider()
    st.subheader("📊 本周经营复盘")
    rev = weekly_review(auto_regen=False)
    if not rev:
        st.warning("暂无数据，请先对工单执行归因")
    else:
        # 待补证队列：直接读 jsonl
        pending = []
        tmap = {}
        tp = os.path.join("data", "tickets_seed.jsonl")
        rp = os.path.join("data", "results.jsonl")
        if os.path.exists(tp):
            with open(tp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        o = json.loads(line)
                        tmap[o["id"]] = o.get("item", "")
        if os.path.exists(rp):
            with open(rp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        o = json.loads(line)
                        if o.get("needs_more_evidence"):
                            pending.append((o.get("id"), tmap.get(o.get("id"), ""), o.get("ask_user")))

        c1, c2, c3 = st.columns(3)
        c1.metric("本周工单", rev["n"])
        c2.metric("估算总损失", f"¥{rev['total_loss']:,.0f}")
        c3.metric("⚠️ 待补证", len(pending))
        st.success("💡 " + rev["conclusion"])

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Top 损失源**")
            trs = "".join(
                f"<tr>"
                f"<td style='padding:6px 10px;border-bottom:1px solid #eee'>{k}</td>"
                f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:center'>{int(c)}</td>"
                f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:right'>¥{v:,.0f}</td>"
                f"</tr>"
                for k, c, v in rev["top_causes"]
            )
            st.markdown(
                "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
                "<thead><tr style='background:#f2f2f2'>"
                "<th style='padding:6px 10px;text-align:left'>二级原因</th>"
                "<th style='padding:6px 10px;text-align:center'>单数</th>"
                "<th style='padding:6px 10px;text-align:right'>损失</th>"
                "</tr></thead><tbody>" + trs + "</tbody></table>",
                unsafe_allow_html=True,
            )
        with cc2:
            st.markdown("**责任方分布**")
            total = sum(rev["party_dist"].values()) or 1
            for k, v in rev["party_dist"].items():
                pct = v / total * 100
                st.markdown(
                    f"<div style='margin:6px 0'>"
                    f"<div style='display:flex;justify-content:space-between;font-size:14px'>"
                    f"<span>{k}</span><span>{v} 单（{pct:.0f}%）</span></div>"
                    f"<div style='background:#eee;border-radius:4px;height:12px'>"
                    f"<div style='width:{pct:.0f}%;background:#4e79a7;height:12px;border-radius:4px'></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

        if pending:
            st.markdown("**🔎 待补证队列（损失被低估的高风险单）**")
            for tid, item, ask in pending:
                st.markdown(f"- **{tid} {item}**：需补充 {ask}")
        if rev["risks"]:
            with st.expander(f"⚠️ 风险单 {len(rev['risks'])} 条"):
                for tid, item, rf in rev["risks"]:
                    st.markdown(f"- **{tid} {item}**：{rf}")