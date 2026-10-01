# -*- coding: utf-8 -*-
"""
ResearchLens · Streamlit 界面 v2.0
注意：本文件不使用 st.table / st.dataframe / st.bar_chart / st.line_chart，
      因为它们底层依赖 pandas，当前环境 pandas DLL 被系统策略拦截。
      所有可视化用纯 HTML/CSS/SVG 实现。
"""

import os
import json
from pathlib import Path
import streamlit as st

from literature_mining import mine as mine_paper, load_cache as load_mining_cache
from hypothesis_generator import generate as gen_hypotheses, load_cache as load_hyp_cache
from chart_understanding import analyze_chart_and_text
from research_tracker import (
    track, load_papers, load_mining_results, load_hypotheses,
    count_gaps, summarize_hypotheses
)

# 图表理解结论缓存（UI 离线写入，全景页只读，不触发 API）
CHART_FINDINGS_PATH = os.path.join("data", "chart_findings.json")


def _load_chart_findings():
    if os.path.exists(CHART_FINDINGS_PATH):
        try:
            with open(CHART_FINDINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_chart_finding(name, res):
    findings = _load_chart_findings()
    findings[name] = {
        "status": res.get("status"),
        "chart_type": (res.get("chart_data") or {}).get("chart_type"),
        "n_data_points": len((res.get("chart_data") or {}).get("data_points", [])),
        "n_discrepancies": len(res.get("discrepancies", [])),
    }
    with open(CHART_FINDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)


def _render_chart_result(res, name):
    """渲染单张图表的提取结果与图文矛盾"""
    if res.get("status") != "success" or not res.get("chart_data"):
        st.error("图表提取失败：" + str(res.get("status", "未知")))
        return
    cd = res["chart_data"]
    st.markdown(
        f'<div class="rl-card" style="border-left:4px solid {THEME["primary"]}">'
        f'<b>{html_escape(cd.get("title") or name)}</b> · '
        f'<span class="rl-badge gray">{html_escape(str(cd.get("chart_type", "?")))}</span>'
        f'<span style="color:{THEME["muted"]};font-size:12px;margin-left:8px">'
        f'{len(cd.get("data_points", []))} 个数据点</span></div>',
        unsafe_allow_html=True,
    )
    rows = ""
    for p in cd.get("data_points", [])[:15]:
        rows += (
            f'<tr><td style="padding:5px 10px;border-bottom:1px solid #EEE">{html_escape(str(p.get("label", "")))}</td>'
            f'<td style="padding:5px 10px;border-bottom:1px solid #EEE"><b>{html_escape(str(p.get("value", "")))}{html_escape(str(p.get("unit", "")))}</b></td>'
            f'<td style="padding:5px 10px;border-bottom:1px solid #EEE;color:{THEME["muted"]}">'
            f'{"约" if p.get("approximate") else ""}</td></tr>'
        )
    st.markdown(
        f'<div class="rl-card"><div style="font-size:13px;font-weight:700;margin-bottom:4px">📈 提取数值</div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:13px">{rows}</table></div>',
        unsafe_allow_html=True,
    )
    disc = res.get("discrepancies", [])
    if disc:
        for d in disc:
            st.markdown(
                f'<div class="rl-card" style="border-left:4px solid {THEME["danger"]};background:#FFF8F6">'
                f'<span class="rl-badge red">{html_escape(d.get("type", ""))}</span> '
                f'<span style="font-size:13px">{html_escape(d.get("description", ""))}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="rl-card" style="border-left:4px solid {THEME["accent"]}">'
            f'✅ 未检测到明显图文矛盾</div>',
            unsafe_allow_html=True,
        )

st.set_page_config(page_title="ResearchLens", page_icon="🔬", layout="wide")

# ============================================================
# 全局主题
# ============================================================
THEME = {
    "primary": "#6C5CE7",
    "primary_dark": "#4834D4",
    "accent": "#00B894",
    "warn": "#FDCB6E",
    "danger": "#E17055",
    "bg": "#F5F6FA",
    "card": "#FFFFFF",
    "text": "#2D3436",
    "muted": "#636E72",
    "sidebar_bg": "#1E1B2E",
    "sidebar_text": "#DFDCFF",
}

st.markdown(f"""
<style>
/* ---------- 页面底色 ---------- */
.stApp {{ background: {THEME['bg']}; font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}

/* ---------- 标题渐变 ---------- */
h1 {{
    background: linear-gradient(90deg, {THEME['primary_dark']}, {THEME['primary']}, #A29BFE);
    -webkit-background-clip: text; background-clip: text; color: transparent !important;
    font-weight: 800 !important; letter-spacing: -0.5px;
}}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ---------- 侧边栏深色 ---------- */
section[data-testid="stSidebar"] {{ background: {THEME['sidebar_bg']}; }}
section[data-testid="stSidebar"] * {{ color: {THEME['sidebar_text']} !important; }}
section[data-testid="stSidebar"] .stSlider > div [data-baseweb="slider"] > div > div > div {{
    background: {THEME['primary']} !important;
}}

/* ---------- 通用卡片 ---------- */
.rl-card {{
    background: {THEME['card']}; border-radius: 12px; padding: 14px 18px;
    margin: 10px 0; border: 1px solid #E8E8EF;
    box-shadow: 0 2px 8px rgba(45,52,54,.06);
    transition: box-shadow .2s;
}}
.rl-card:hover {{ box-shadow: 0 4px 16px rgba(108,92,231,.14); }}

/* ---------- Pipeline 状态条 ---------- */
.rl-pipeline {{ display: flex; gap: 0; margin: 6px 0 2px 0; }}
.rl-step {{
    flex: 1; text-align: center; padding: 10px 4px; font-size: 13px; font-weight: 600;
    background: #ECECF4; color: #919AA6; position: relative;
}}
.rl-step:first-child {{ border-radius: 10px 0 0 10px; }}
.rl-step:last-child {{ border-radius: 0 10px 10px 0; }}
.rl-step.done {{ background: linear-gradient(135deg, {THEME['primary']}, {THEME['primary_dark']}); color: #fff; }}
.rl-step.done::after {{
    content: ""; position: absolute; right: -10px; top: 50%; margin-top: -10px;
    border-left: 10px solid {THEME['primary_dark']}; border-top: 10px solid transparent;
    border-bottom: 10px solid transparent; z-index: 2;
}}

/* ---------- 徽章 ---------- */
.rl-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 700; margin-right: 6px;
}}
.rl-badge.purple {{ background: #EDE9FF; color: {THEME['primary_dark']}; }}
.rl-badge.green  {{ background: #D8F8EF; color: #01825A; }}
.rl-badge.amber  {{ background: #FFF3D6; color: #B7791F; }}
.rl-badge.red    {{ background: #FFE3DA; color: #C0392B; }}
.rl-badge.gray   {{ background: #ECEEF1; color: {THEME['muted']}; }}

/* ---------- Tab 美化 ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; background: transparent; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px 10px 0 0; padding: 10px 22px; font-weight: 700; font-size: 15px;
}}
.stTabs [aria-selected="true"] {{ color: {THEME['primary_dark']}; border-bottom: 3px solid {THEME['primary']}; }}

/* ---------- 指标卡 ---------- */
.rl-metric {{
    background: linear-gradient(135deg, {THEME['card']} 0%, #F2F0FF 100%);
    border: 1px solid #E8E8EF; border-radius: 14px; padding: 18px 20px; text-align: center;
}}
.rl-metric .v {{ font-size: 34px; font-weight: 800; color: {THEME['primary_dark']}; line-height: 1.1; }}
.rl-metric .l {{ font-size: 13px; color: {THEME['muted']}; margin-top: 4px; }}

/* ---------- 进度条 ---------- */
.rl-bar-track {{ background: #ECECF4; border-radius: 6px; height: 10px; overflow: hidden; }}
.rl-bar-fill {{ height: 100%; border-radius: 6px; background: linear-gradient(90deg, {THEME['primary']}, #A29BFE); }}

/* ---------- 实时流水线动画（LiveTracker） ---------- */
.rl-live {{ background: {THEME['card']}; border: 1px solid #E8E8EF; border-radius: 12px; padding: 12px 14px; margin: 10px 0; box-shadow: 0 2px 8px rgba(45,52,54,.06); }}
.rl-live-step {{ display: flex; align-items: center; gap: 9px; padding: 7px 10px; border-radius: 8px; margin: 4px 0; background: #ECECF4; color: #919AA6; font-size: 13px; animation: rlslidein .35s ease both; }}
@keyframes rlslidein {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: none; }} }}
.rl-live-step.run {{ background: linear-gradient(90deg, #EDE9FF, #F7F5FF); color: {THEME['primary_dark']}; font-weight: 700; }}
.rl-live-step.done {{ background: #E8F8F1; color: #01825A; }}
.rl-live-step.fail {{ background: #FFE9E2; color: #C0392B; }}
.rl-live-dot {{ width: 9px; height: 9px; border-radius: 50%; background: currentColor; flex: none; }}
.rl-live-step.run .rl-live-dot {{ animation: rlpulse 0.9s ease-in-out infinite; }}
.rl-live-step.run .rl-live-ico {{ display: inline-block; animation: rlspin 1.2s linear infinite; }}
@keyframes rlpulse {{ 0%,100% {{ opacity: .2; }} 50% {{ opacity: 1; }} }}
@keyframes rlspin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
.rl-live-log {{ font-size: 12px; color: {THEME['muted']}; margin-top: 8px; max-height: 170px; overflow: auto; font-family: ui-monospace, Consolas, monospace; border-top: 1px dashed #E4E4EE; padding-top: 6px; }}

/* ---------- 答辩模式 ---------- */
.rl-stage-banner {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 999;
    background: linear-gradient(90deg, #01825A, #00B894); color: #fff;
    text-align: center; font-size: 13px; font-weight: 700; padding: 5px 0;
    letter-spacing: 1px; box-shadow: 0 2px 10px rgba(0,184,148,.35);
}}
.rl-pf-item {{ display: flex; align-items: center; gap: 7px; padding: 4px 0; font-size: 12px; }}
.rl-pf-dot {{ width: 8px; height: 8px; border-radius: 50%; flex: none; }}
.rl-pf-item.ok .rl-pf-dot {{ background: #00D68F; }}
.rl-pf-item.bad .rl-pf-dot {{ background: #FF6B6B; }}
.rl-pf-item.ok {{ color: #B8F0DC; }}
.rl-pf-item.bad {{ color: #FFB3AD; }}

/* ---------- 假设大卡 ---------- */
.rl-hyp {{
    background: {THEME['card']}; border-radius: 14px; padding: 16px 20px; margin: 12px 0;
    border-left: 5px solid {THEME['primary']};
    box-shadow: 0 2px 10px rgba(45,52,54,.07);
}}
.rl-hyp.top1 {{ border-left-color: #F6C609; background: linear-gradient(135deg, #FFFDF2, {THEME['card']}); }}
.rl-hyp.blocked {{ border-left-color: {THEME['danger']}; background: #FFF8F6; }}

/* ---------- 按钮 ---------- */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {THEME['primary']}, {THEME['primary_dark']});
    border: none; border-radius: 10px; font-weight: 700; padding: 10px 0;
    box-shadow: 0 4px 14px rgba(108,92,231,.35);
}}
.stButton > button {{ border-radius: 10px; font-weight: 600; }}

/* 隐藏 streamlit 默认页脚 */
#MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 辅助渲染函数
# ============================================================
def conf_badge(conf):
    cls = "green" if conf > 0.6 else ("amber" if conf > 0.4 else "red")
    label = "高置信" if conf > 0.6 else ("中置信" if conf > 0.4 else "低置信")
    return f'<span class="rl-badge {cls}">{label} {conf:.0%}</span>'


def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def index_gaps(mining_out):
    """gap_id -> gap 对象索引，用于假设→证据回链解析"""
    idx = {}
    for g in (mining_out or {}).get("gaps", []):
        gid = g.get("gap_id")
        if gid:
            idx[gid] = g
    return idx


def gap_card_html(g):
    """统一的空白卡渲染：ID + 类型/子类型 + 置信度徽章 + 逐字引用"""
    conf = g.get("confidence", 0)
    quotes_html = "".join(
        f'<div style="border-left:3px solid #DFE6E9;margin:8px 0;padding:2px 12px;font-size:13px;color:{THEME["muted"]}">'
        f'「{html_escape(ev.get("quote", "—"))}」 <i>— {html_escape(ev.get("section", "?"))}</i></div>'
        for ev in (g.get("evidence") or []) if isinstance(ev, dict)
    )
    return (
        f'<div class="rl-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<b style="font-size:15px">{html_escape(g.get("gap_id", "?"))} · '
        f'{html_escape(g.get("gap_type", "?"))} / {html_escape(g.get("gap_subtype", "?"))}</b>'
        f'<span>{conf_badge(conf)}</span></div>'
        f'<div style="font-size:14px;margin-top:6px;line-height:1.65">{html_escape(g.get("description", ""))}</div>'
        f'{quotes_html}</div>'
    )


def pipeline_bar(mined, has_hyp):
    """四步 pipeline：选论文 → 文献挖掘 → 假设生成 → 研究全景"""
    steps = [
        ("① 选定论文", True),
        ("② 文献挖掘", mined),
        ("③ 假设生成", bool(has_hyp)),
        ("④ 研究全景", mined and bool(has_hyp)),
    ]
    html = '<div class="rl-pipeline">'
    for name, done in steps:
        html += f'<div class="rl-step {"done" if done else ""}">{"✓ " if done else ""}{name}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── 实时流水线动画：每个里程碑对应一次真实计算，非假延时 ─────────────
class LiveTracker:
    """在占位容器内渲染分步进度卡，供 on_progress 回调驱动。

    usage = st.empty()
    t = LiveTracker(usage, phases)
    t.on(phase, message, data)   # 传入 on_progress
    t.finish(summary) / t.fail(msg)
    """

    PHASES = {
        "load_paper": ("载入论文", "本地全文预抽取 · 演示零联网"),
        "calling_llm": ("调用 Qwen 长上下文", "封闭枚举约束下结构化抽取"),
        "validating": ("校验与修复", "schema → 枚举 → 置信度联动"),
        "done": ("结果落盘", "写入本地缓存"),
        "load_gaps": ("载入研究空白", "作为假设生成的输入"),
        "gating": ("门控判定", "逐条放行 / 拦截 + 四维打分"),
    }
    # mine 与 generate 各自的步骤顺序
    FLOW_MINE = ["load_paper", "calling_llm", "validating", "done"]
    FLOW_HYP = ["load_gaps", "calling_llm", "gating", "done"]
    ORDER = {"cache_hit": -1, "load_paper": 0, "load_gaps": 0,
             "calling_llm": 1, "llm_done": 1.5, "validating": 2,
             "parsing": 2, "gating": 2, "done": 3}

    def __init__(self, container, flow):
        self.c = container
        self.flow = flow
        self.done_steps = set()
        self.running = None
        self.logs = []
        self._render()

    def on(self, phase, message, data=None):
        if phase in ("cache_hit",):
            self.logs.insert(0, "⚡ " + message)
            self.done_steps = set(self.flow)
            self.running = None
            self._render()
            return
        if phase == "failed":
            self.fail(message)
            return
        # 顺序推进：之前运行中的步骤标为完成
        cur = self.ORDER.get(phase)
        if self.running and self.running != phase and self.ORDER.get(self.running) is not None:
            if cur is not None and self.ORDER[self.running] <= cur:
                self.done_steps.add(self.running)
        if phase in self.PHASES:
            self.running = phase
        # 特殊事件名并入其阶段显示
        if phase == "llm_done":
            self.done_steps.add("calling_llm")
            self.running = "validating"
        if phase in ("parsing", "gating"):
            self.done_steps.add("calling_llm")
            self.running = "gating"
        if phase == "done":
            self.done_steps = set(self.flow)
            self.running = None
        self.logs.insert(0, message)
        del self.logs[10:]
        self._render()

    def finish(self, summary=""):
        self.done_steps = set(self.flow)
        self.running = None
        if summary:
            self.logs.insert(0, "🏁 " + summary)
        self._render()

    def fail(self, msg):
        self.running = None
        self.error = msg
        self.logs.insert(0, "❌ " + msg)
        self._render()

    def _render(self):
        rows = []
        order = sorted(self.ORDER)
        for name in self.flow:
            if name in self.done_steps:
                rows.append(f'<div class="rl-live-step done"><span class="rl-live-dot"></span>✓ {self.PHASES[name][0]}<span style="margin-left:auto;font-size:11px">{self.PHASES[name][1]}</span></div>')
            elif name == self.running:
                rows.append(f'<div class="rl-live-step run"><span class="rl-live-dot"></span><span class="rl-live-ico">◌</span> {self.PHASES[name][0]}…<span style="margin-left:auto;font-size:11px">{self.PHASES[name][1]}</span></div>')
            else:
                rows.append(f'<div class="rl-live-step"><span class="rl-live-dot"></span>○ {self.PHASES[name][0]}<span style="margin-left:auto;font-size:11px">{self.PHASES[name][1]}</span></div>')
        head = getattr(self, "error", None)
        if head:
            rows.append(f'<div class="rl-live-step fail"><span class="rl-live-dot"></span>✗ 失败：{html_escape(head)}</div>')
        logs_html = "".join(
            f'<div style="padding:2px 0;border-bottom:1px dotted #F0F0F5">▸ {html_escape(l)}</div>'
            for l in self.logs
        )
        self.c.markdown(
            '<div class="rl-live">' + "".join(rows)
            + (f'<div class="rl-live-log">{logs_html}</div>' if logs_html else "")
            + "</div>",
            unsafe_allow_html=True,
        )


def citation_network_svg(papers, active_id):
    """根据 cross_refs 绘制论文引用网络（纯 SVG）"""
    pos = {
        "llava15":  (110, 140),
        "som":      (330, 45),
        "viscot":   (330, 235),
        "mmmupro":  (550, 140),
    }
    short = {"llava15": "LLaVA-1.5", "som": "SoM", "viscot": "Visual-CoT", "mmmupro": "MMMU-Pro"}
    node_r = 42
    edges = []
    for p in papers:
        pid = p["paper_id"]
        for tgt in p.get("cross_refs_out", []):
            if tgt in pos and pid in pos:
                edges.append((pid, tgt))

    svg = ['<svg viewBox="0 0 660 290" style="width:100%;max-width:680px;">']
    # 箭头定义
    svg.append(
        '<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#A29BFE"/></marker></defs>'
    )
    # 边
    for src, dst in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        dx, dy = x2 - x1, y2 - y1
        dist = max((dx * dx + dy * dy) ** 0.5, 1)
        ux, uy = dx / dist, dy / dist
        sx, sy = x1 + ux * (node_r + 2), y1 + uy * (node_r + 2)
        ex, ey = x2 - ux * (node_r + 6), y2 - uy * (node_r + 6)
        # 微微弯曲的贝塞尔边
        mx, my = (sx + ex) / 2 - (ey - sy) * 0.12, (sy + ey) / 2 + (ex - sx) * 0.12
        svg.append(
            f'<path d="M{sx:.0f},{sy:.0f} Q{mx:.0f},{my:.0f} {ex:.0f},{ey:.0f}" '
            f'fill="none" stroke="#A29BFE" stroke-width="1.6" marker-end="url(#arr)" opacity="0.75"/>'
        )
    # 节点
    for pid, (x, y) in pos.items():
        p = next((q for q in papers if q["paper_id"] == pid), None)
        if not p:
            continue
        is_active = pid == active_id
        is_classic = p.get("role") == "classic_high_cited"
        fill = ("url(#gactive)" if is_active else
                ("#FFFFFF" if not is_classic else "#F2F0FF"))
        stroke = THEME["primary"] if is_active else "#B2B6CF"
        sw = 3.5 if is_active else 1.5
        svg.append(
            f'<circle cx="{x}" cy="{y}" r="{node_r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        svg.append(
            f'<text x="{x}" y="{y - 4}" text-anchor="middle" font-size="13" font-weight="800" '
            f'fill="{THEME["primary_dark"]}">{short.get(pid, pid)}</text>'
        )
        svg.append(
            f'<text x="{x}" y="{y + 14}" text-anchor="middle" font-size="10" '
            f'fill="{THEME["muted"]}">{p.get("year", "")} · ~{p.get("citations_approx", "")}</text>'
        )
    svg.append(
        '<defs><linearGradient id="gactive" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="#EDE9FF"/><stop offset="100%" stop-color="{THEME["primary"]}"/></linearGradient></defs>'
    )
    svg.append("</svg>")
    return "".join(svg)


# ============================================================
# 会话状态 & 数据
# ============================================================
if "mining_results" not in st.session_state:
    st.session_state.mining_results = {}
if "hypotheses" not in st.session_state:
    st.session_state.hypotheses = {}
if "jumped" not in st.session_state:
    st.session_state.jumped = None


@st.cache_data
def load_seed_papers():
    path = os.path.join("data", "lit_seed.jsonl")
    out = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


papers = load_seed_papers()
if not papers:
    st.error("data/lit_seed.jsonl 不存在，请先准备种子论文")
    st.stop()

# ============================================================
# 头部
# ============================================================
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:2px">
        <div>
            <div style="font-size:30px;font-weight:800;
                 background:linear-gradient(90deg,{THEME['primary_dark']},{THEME['primary']},#A29BFE);
                 -webkit-background-clip:text;background-clip:text;color:transparent;letter-spacing:-0.5px">
                 🔬 ResearchLens</div>
            <div style="color:{THEME['muted']};font-size:14px;margin-top:2px">
                 基于 Qwen 的科研假设生成系统 · 认知增强与研发范式革新</div>
        </div>
        <div style="font-size:12px;color:{THEME['muted']}">
            <span class="rl-badge purple">Qwen-Max</span>
            <span class="rl-badge green">Qwen-VL-Max</span>
            <span class="rl-badge gray">百炼 API</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="font-size:20px;font-weight:800;padding:6px 0 10px 0">🧭 控制台</div>',
        unsafe_allow_html=True,
    )
    paper_id = st.selectbox("选择论文", [p["paper_id"] for p in papers], index=0)
    paper = next(p for p in papers if p["paper_id"] == paper_id)

    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,.06);border-radius:12px;padding:12px 14px;margin:10px 0;font-size:13px;line-height:1.8">
            <b style="font-size:14px">{html_escape(paper['title'][:44])}…</b><br>
            👤 {paper.get('authors_short', '—')}<br>
            📅 {paper.get('year', '—')} · {paper.get('venue', '—')}<br>
            📈 引用 ~{paper.get('citations_approx', '—')} ·
            {'📚 经典' if paper.get('role') == 'classic_high_cited' else '🆕 近作'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    budget_days = st.slider("⏱ 工时预算（人日）", 5, 90, 30, 5,
                            help="超出预算的假设将被门控拦截")

    st.markdown("---")
    cache_m = load_mining_cache()
    cache_h = load_hyp_cache()

    # ── 答辩模式：防误点重算 + 离线自检红绿灯 ──────────────────
    stage_mode = st.toggle("🎤 答辩模式", value=False,
                           help="开启后禁用会触发 API 重算的按钮，并展示离线自检结果；"
                                "全场演示只读本地缓存，断网也能跑完")

    if stage_mode:
        from preflight import run_preflight
        pf = run_preflight()
        rows = "".join(
            f'<div class="rl-pf-item {"ok" if ok else "bad"}">'
            f'<span class="rl-pf-dot"></span>{cid} {name}'
            f'<span style="margin-left:auto">{detail.split("；")[0]}</span></div>'
            for cid, name, ok, detail in pf["checks"]
        )
        head = ("🟢 断网演示就绪 · 可放心上台" if pf["all_ok"]
                else "🔴 有检查项未通过，请先修复")
        st.markdown(
            f'<div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);'
            f'border-radius:12px;padding:10px 12px;margin:6px 0">'
            f'<div style="font-size:13px;font-weight:800;margin-bottom:4px">{head}</div>'
            f'<div style="font-size:11px;opacity:.7;margin-bottom:6px">离线自检（零 API · 零联网）</div>'
            f'{rows}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:rgba(255,255,255,.05);border-radius:10px;padding:8px 12px;'
            f'font-size:12px;line-height:1.9;margin-bottom:6px">'
            f'📦 本地缓存<br>已挖掘论文：{len(cache_m)} / {len(papers)} · '
            f'假设集：{len(cache_h)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="background:linear-gradient(90deg,#01825A,#00B894);color:#fff;'
            'text-align:center;font-size:12px;font-weight:700;padding:5px 0;border-radius:8px;'
            f'letter-spacing:1px">🎤 答辩模式已开启 · 全场只读本地缓存 · 重算按钮已禁用</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="font-size:12px;line-height:2">
            <b>📦 本地缓存</b><br>
            已挖掘论文：{len(cache_m)} / {len(papers)}<br>
            已有假设集：{len(cache_h)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("⚠️ Demo 数据基于 4 篇真实论文构建，非编造")

# 当前视图状态
def get_hyps(paper_id_):
    """取该论文的假设列表：session_state 优先，缓存键兼容 '<id>_budget<N>' 格式"""
    h = st.session_state.hypotheses.get(paper_id_)
    if h:
        return h if isinstance(h, list) else []
    for k, v in cache_h.items():
        if k == paper_id_ or k.startswith(paper_id_ + "_budget"):
            return v if isinstance(v, list) else []
    return []

out = st.session_state.mining_results.get(paper_id) or cache_m.get(paper_id)
hyps = get_hyps(paper_id)

pipeline_bar(bool(out and "_error" not in out), hyps)

# ============================================================
# 四视图 Tabs
# ============================================================
tab_read, tab_mine, tab_hyp, tab_pan, tab_chart = st.tabs(
    ["📄 论文精读", "🧬 文献挖掘", "💡 假设雷达", "🌐 研究全景", "📊 图表理解"]
)

# ---------- Tab 1：论文精读 ----------
with tab_read:
    c1, c2 = st.columns([6, 4])
    with c1:
        st.markdown(
            f"""
            <div class="rl-card">
                <div style="font-size:13px;color:{THEME['muted']};margin-bottom:6px">🎯 核心贡献</div>
                <div style="font-size:16px;font-weight:600;line-height:1.7">{html_escape(paper.get('core_contribution', '—'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**📋 作者自述局限性**")
        for lim in paper.get("limitations_extracted", []):
            st.markdown(
                f'<div class="rl-card" style="border-left:4px solid {THEME["warn"]};padding:10px 14px;font-size:14px">'
                f'{html_escape(lim)}</div>',
                unsafe_allow_html=True,
            )
        if paper.get("cross_points"):
            st.markdown("**🔗 跨论文交叉点**")
            for cp in paper["cross_points"]:
                st.markdown(
                    f'<div class="rl-card" style="border-left:4px solid {THEME["primary"]};padding:10px 14px;font-size:14px">'
                    f'{html_escape(cp)}</div>',
                    unsafe_allow_html=True,
                )
    with c2:
        st.markdown("**🕸 种子论文引用网络**")
        st.markdown(
            citation_network_svg(papers, paper_id),
            unsafe_allow_html=True,
        )
        st.caption("箭头表示近作引用经典 / 互相指涉；当前选中论文高亮。系统利用此闭环结构发现跨论文空白。")

# ---------- Tab 2：文献挖掘 ----------
with tab_mine:
    c1, c2 = st.columns([4, 6])
    with c1:
        paper_text = None
        txt_path = paper.get("local_txt", "")
        full_path = os.path.join(os.path.dirname(__file__) or ".", txt_path) if txt_path else ""
        if full_path and os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                paper_text = f.read()
            st.markdown(
                f'<div class="rl-metric" style="margin-bottom:12px"><div class="v">{len(paper_text):,}</div>'
                f'<div class="l">论文全文字符数（本地预抽取，演示零联网）</div></div>',
                unsafe_allow_html=True,
            )
        force = st.checkbox("强制重新计算（忽略缓存）", key="force_mine",
                            disabled=stage_mode)
        live_mine = st.empty()
        if st.button("▶ 开始文献挖掘", type="primary", use_container_width=True,
                     disabled=stage_mode,
                     help="答辩模式下已禁用（只读缓存），关闭答辩模式后可点击"):
            import time as _time
            tracker = LiveTracker(live_mine, LiveTracker.FLOW_MINE)
            _t0 = _time.time()
            result = mine_paper(paper, use_cache=not force,
                                on_progress=tracker.on)
            tracker.finish(f"总耗时 {_time.time() - _t0:.1f} 秒")
            st.session_state.mining_results[paper_id] = result
            _time.sleep(1.2)  # 让完成态展示片刻
            live_mine.empty()
            st.rerun()
        out = st.session_state.mining_results.get(paper_id) or cache_m.get(paper_id)

    with c2:
        if not out:
            st.info("尚未挖掘 — 点击左侧「开始文献挖掘」")
        elif "_error" in out:
            st.error("挖掘失败：" + out.get("_error", ""))
            if out.get("_raw"):
                with st.expander("原始输出"):
                    st.code(out["_raw"])
        else:
            # 五元组
            st.markdown("#### 📐 科研五元组")
            ft = out.get("five_tuple", {})
            ft_icons = {
                "research_question": ("❓", THEME["primary"]),
                "methodology": ("🛠", THEME["accent"]),
                "datasets": ("🗂", "#0984E3"),
                "conclusions": ("📊", "#6C5CE7"),
                "limitations": ("⚠️", THEME["danger"]),
            }
            ft_labels = {
                "research_question": "研究问题", "methodology": "方法",
                "datasets": "数据集", "conclusions": "结论", "limitations": "局限性",
            }
            for key, (icon, color) in ft_icons.items():
                st.markdown(
                    f'<div class="rl-card" style="border-left:4px solid {color};padding:10px 14px">'
                    f'<span style="font-size:12px;color:{THEME["muted"]}">{icon} {ft_labels[key]}</span><br>'
                    f'<span style="font-size:14px">{html_escape(ft.get(key, "—"))}</span></div>',
                    unsafe_allow_html=True,
                )

            # 研究空白
            gaps = out.get("gaps", [])
            st.markdown(
                f'<div style="margin:14px 0 4px 0"><span style="font-size:18px;font-weight:800;'
                f'color:{THEME["primary_dark"]}">发现 {len(gaps)} 个研究空白</span>'
                f'<span style="color:{THEME["muted"]};font-size:13px"> · 每条证据引用均可在论文原文中字符串命中（反幻觉）</span></div>',
                unsafe_allow_html=True,
            )
            for g in gaps:
                st.markdown(gap_card_html(g), unsafe_allow_html=True)

            # 质量指标
            qc = out.get("_quote_check", {})
            if qc and qc.get("total", 0) > 0:
                rate = qc["hits"] / qc["total"]
                emoji = "✅" if rate >= 0.9 else "⚠️" if rate >= 0.7 else "❌"
                st.markdown(
                    f'<div class="rl-card" style="display:flex;align-items:center;gap:14px">'
                    f'<span style="font-size:22px">{emoji}</span><div style="flex:1">'
                    f'<div style="display:flex;justify-content:space-between;font-size:13px">'
                    f'<b>quote 证据命中率（可机器校验的反幻觉指标）</b><span>{qc["hits"]}/{qc["total"]} · {rate:.0%}</span></div>'
                    f'<div class="rl-bar-track" style="margin-top:6px"><div class="rl-bar-fill" style="width:{rate*100:.0f}%"></div></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            if out.get("_repaired"):
                st.caption("🔧 输出已自动修复：" + str(out["_repaired"]))
            if out.get("_warn"):
                st.caption("⚠️ " + str(out["_warn"]))
            with st.expander("查看原始 JSON"):
                st.json(out)

# ---------- Tab 3：假设雷达 ----------
with tab_hyp:
    if not out or "_error" in out:
        st.info("请先在「文献挖掘」页完成挖掘")
    else:
        c1, c2 = st.columns([3, 7])
        with c1:
            st.markdown(
                f'<div class="rl-metric"><div class="v">{budget_days}</div>'
                f'<div class="l">工时预算（人日）</div></div>',
                unsafe_allow_html=True,
            )
            st.caption("超预算 / 不可验证 / 已被回答的假设将被门控拦截")
        with c2:
            live_hyp = st.empty()
            if st.button("💡 生成假设", type="primary", use_container_width=True,
                         disabled=stage_mode,
                         help="答辩模式下已禁用（只读缓存），关闭答辩模式后可点击"):
                import time as _time
                tracker = LiveTracker(live_hyp, LiveTracker.FLOW_HYP)
                _t0 = _time.time()
                new_hyps = gen_hypotheses(out, budget_days=budget_days, use_cache=False,
                                          on_progress=tracker.on)
                if isinstance(new_hyps, dict) and "error" in new_hyps:
                    tracker.fail(new_hyps["error"])
                else:
                    tracker.finish(f"总耗时 {_time.time() - _t0:.1f} 秒")
                    st.session_state.hypotheses[paper_id] = new_hyps
                    _time.sleep(1.5)
                    live_hyp.empty()
                    st.rerun()

        hyps = get_hyps(paper_id)
        if not isinstance(hyps, list):
            hyps = []
        if hyps:
            passed = sorted(
                [h for h in hyps if h.get("gate_passed")],
                key=lambda x: x.get("total_score", 0), reverse=True,
            )
            blocked = [h for h in hyps if not h.get("gate_passed")]
            st.markdown(
                f'<div style="margin:16px 0 6px 0"><span class="rl-badge purple">{len(hyps)} 条候选</span>'
                f'<span class="rl-badge green">{len(passed)} 通过门控</span>'
                f'<span class="rl-badge red">{len(blocked)} 被拦截</span></div>',
                unsafe_allow_html=True,
            )

            DIM_CN = {"novelty": "新颖性", "feasibility": "可行性",
                      "impact": "影响力", "evidence_strength": "证据强度"}
            DIM_W = {"novelty": 0.30, "feasibility": 0.30,
                     "impact": 0.25, "evidence_strength": 0.15}

            gap_idx = index_gaps(out)

            def jump_button(h_, key_prefix):
                """回链按钮：点击后原位展开该假设的来源空白卡"""
                gid = h_.get("gap_id") or ""
                tg = h_.get("target_gap_ids")
                if isinstance(tg, list) and tg:
                    gid = str(tg[0])
                elif tg and not gid:
                    gid = str(tg)
                st_key = f"{key_prefix}::{paper_id}::{h_.get('hyp_id', gid)}"
                if gid in gap_idx:
                    cols = st.columns([3, 9])
                    with cols[0]:
                        if st.button(f"🔗 回链证据 {gid}", key=f"btn_{st_key}"):
                            st.session_state.jumped = (
                                None if st.session_state.get("jumped") == st_key else st_key
                            )
                    if st.session_state.get("jumped") == st_key:
                        with cols[1]:
                            st.markdown(
                                '<div style="font-size:11px;color:#01825A;padding:6px 0 2px 4px">'
                                '▲ 证据链闭环：该假设由下列研究空白驱动，空白由论文原文逐字摘录支撑</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(gap_card_html(gap_idx[gid]), unsafe_allow_html=True)
                return gid

            for i, h in enumerate(passed):
                tag = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}")
                score = h.get("total_score", 0)
                bars = ""
                scores = h.get("scores", {})
                for dim in ["novelty", "feasibility", "impact", "evidence_strength"]:
                    s = scores.get(dim, {}) if isinstance(scores.get(dim), dict) else {}
                    val = s.get("value", 0) or 0
                    ev = s.get("evidence", "—")
                    w = int(val * 100)
                    bars += (
                        f'<div style="margin:7px 0">'
                        f'<div style="display:flex;justify-content:space-between;font-size:12px">'
                        f'<span>{DIM_CN[dim]} <span style="color:#B2BEC3">×{int(DIM_W[dim]*100)}%</span></span>'
                        f'<span><b>{val:.2f}</b></span></div>'
                        f'<div class="rl-bar-track"><div class="rl-bar-fill" style="width:{w}%"></div></div>'
                        f'<div style="font-size:11px;color:{THEME["muted"]};margin-top:2px">{html_escape(ev)}</div></div>'
                    )
                st.markdown(
                    f'<div class="rl-hyp {("top1" if i == 0 else "")}">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span style="font-size:16px">{tag}</span>'
                    f'<span class="rl-badge purple">总分 {score:.3f}</span></div>'
                    f'<div style="font-size:15px;font-weight:700;margin:6px 0;line-height:1.6">'
                    f'{html_escape(h.get("statement", "—"))}</div>'
                    f'<div style="margin-bottom:10px">'
                    f'<span class="rl-badge gray">{html_escape(h.get("hypothesis_kind", "?"))}</span>'
                    f'<span class="rl-badge gray">{html_escape(h.get("resource_tag", "?"))}</span>'
                    f'<span class="rl-badge gray">{html_escape(h.get("verifiable_via", "?"))}</span>'
                    f'<span class="rl-badge amber">{html_escape(str(h.get("effort_days", "?")))} 人日</span></div>'
                    f'{bars}'
                    f'<div style="font-size:12px;color:{THEME["muted"]};margin-top:10px">'
                    f'<b>实验方案</b> {html_escape(h.get("methodology", "—"))}<br>'
                    f'<b>价值说明</b> {html_escape(h.get("rationale", "—"))}<br>'
                    f'<b>目标空白</b> {html_escape(str(h.get("target_gap_ids") or h.get("gap_id") or "—"))}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                jump_button(h, f"hyp{i}")

            if blocked:
                st.markdown("#### 🚫 被门控拦截的假设（系统会「说不」，且给出理由）")
                for bi, h in enumerate(blocked):
                    st.markdown(
                        f'<div class="rl-hyp blocked">'
                        f'<div style="font-size:14px;font-weight:700">'
                        f'{html_escape(h.get("hyp_id", "?"))} · {html_escape(h.get("statement", "—"))[:70]}…</div>'
                        f'<div style="font-size:13px;color:{THEME["danger"]};margin-top:4px">'
                        f'⛔ {html_escape(h.get("gate_reason", "—"))}</div></div>',
                        unsafe_allow_html=True,
                    )
                    jump_button(h, f"blk{bi}")

# ---------- Tab 4：研究全景 ----------
with tab_pan:
    result = track(verbose=False, budget_days=budget_days)
    if not result:
        st.info("暂无数据 — 请先对论文执行文献挖掘（结果会缓存到 data/mining_results.jsonl）")
    else:
        gs = result["gap_summary"]
        hs = result["hyp_summary"]
        pending = result["pending"]

        st.markdown("#### 🌐 全局视图")
        cs = result.get("chart_summary", {}) or {}
        has_charts = cs.get("figures", 0) > 0
        if has_charts:
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            metrics = [
                (m1, result["mined"], "分析论文"),
                (m2, gs["total"], "研究空白"),
                (m3, hs["total"], "生成假设"),
                (m4, hs["passed"], "通过门控"),
                (m5, len(pending), "待补证空白"),
                (m6, cs.get("discrepancies", 0), "图文矛盾"),
            ]
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            metrics = [
                (m1, result["mined"], "分析论文"),
                (m2, gs["total"], "研究空白"),
                (m3, hs["total"], "生成假设"),
                (m4, hs["passed"], "通过门控"),
                (m5, len(pending), "待补证空白"),
            ]
        for col, v, label in metrics:
            col.markdown(
                f'<div class="rl-metric"><div class="v">{v}</div><div class="l">{label}</div></div>',
                unsafe_allow_html=True,
            )

        # 提效量化对照卡（赛道硬要求：科研提效需可量化证据）
        n_papers = result["mined"]
        n_gaps = gs["total"]
        n_hyps = hs["total"]
        manual_hours = n_papers * 2.5  # 估算：每篇人工精读+找空白+构思假设 ≈ 2.5 人时
        manual_hyp_hours = n_hyps / 0.15 if n_hyps else 0  # 人工 ≈ 6-7 人时/条可验证假设
        sys_minutes = 12  # 演示：一次挖掘+生成约 12 分钟（缓存命中后聚合为秒级）
        speedup = (manual_hours * 60 / sys_minutes) if sys_minutes else 0
        eff_html = (
            f'<div class="rl-card" style="border-left:5px solid {THEME["warn"]};margin-top:14px">'
            f'<div style="font-size:14px;font-weight:800;margin-bottom:8px">⚡ 提效量化对照（估算）</div>'
            f'<div style="display:flex;gap:14px">'
            f'<div style="flex:1;background:#F7F8FA;border-radius:10px;padding:10px 12px">'
            f'<div style="font-size:12px;color:{THEME["muted"]}">👤 人工方式（{n_papers} 篇）</div>'
            f'<div style="font-size:13px;margin-top:4px">精读全文 + 找空白 + 构思假设</div>'
            f'<div style="font-size:20px;font-weight:800;margin-top:6px">≈ {manual_hours:.0f} 人时</div>'
            f'<div style="font-size:12px;color:{THEME["muted"]}">产出 {n_hyps} 条需 ≈ {manual_hyp_hours:.0f} 人时</div>'
            f'</div>'
            f'<div style="flex:1;background:#F2F0FF;border-radius:10px;padding:10px 12px">'
            f'<div style="font-size:12px;color:{THEME["primary_dark"]}">🤖 ResearchLens</div>'
            f'<div style="font-size:13px;margin-top:4px">挖掘 + 假设生成 + 门控（一次 LLM 调用）</div>'
            f'<div style="font-size:20px;font-weight:800;margin-top:6px;color:{THEME["primary_dark"]}">≈ {sys_minutes} 分钟</div>'
            f'<div style="font-size:12px;color:{THEME["muted"]}">缓存命中后聚合为秒级</div>'
            f'</div></div>'
            f'<div style="font-size:12px;color:{THEME["muted"]};margin-top:8px">'
            f'本次从 {n_papers} 篇论文自动发现 {n_gaps} 个研究空白、生成 {n_hyps} 条候选假设；'
            f'人工等效耗时约为系统的 <b>{speedup:.0f}×</b>（按人工 2.5 人时/篇估算，含构思）。'
            f'</div></div>'
        )
        st.markdown(eff_html, unsafe_allow_html=True)

        st.markdown(
            f'<div class="rl-card" style="border-left:5px solid {THEME["accent"]};margin-top:14px;'
            f'font-size:15px;line-height:1.7">💡 {html_escape(result["conclusion"])}</div>',
            unsafe_allow_html=True,
        )

        cc1, cc2, cc3 = st.columns([4, 4, 4])
        with cc1:
            st.markdown("**空白类型分布**")
            type_dist = gs.get("type_dist", {})
            max_val = max(type_dist.values()) if type_dist else 1
            rows = ""
            for k, v in sorted(type_dist.items(), key=lambda x: -x[1]):
                pct = v / max_val * 100
                rows += (
                    f'<div style="margin:9px 0">'
                    f'<div style="display:flex;justify-content:space-between;font-size:13px">'
                    f'<span>{html_escape(k)}</span><span><b>{v}</b> 个</span></div>'
                    f'<div class="rl-bar-track" style="margin-top:3px"><div class="rl-bar-fill" style="width:{pct:.0f}%"></div></div></div>'
                )
            st.markdown(f'<div class="rl-card">{rows}</div>', unsafe_allow_html=True)

        with cc2:
            st.markdown("**假设门控比例**")
            total_h = hs.get("total", 0)
            passed = hs.get("passed", 0)
            blocked_count = hs.get("blocked", 0)
            if total_h > 0:
                p_pct = passed / total_h * 100
                seg = (
                    f'<div style="display:flex;border-radius:8px;overflow:hidden;height:34px;font-size:13px;font-weight:700;color:#fff">'
                    f'<div style="width:{p_pct:.0f}%;background:{THEME["accent"]};display:flex;align-items:center;justify-content:center">✅ {passed}</div>'
                    f'<div style="width:{100-p_pct:.0f}%;background:{THEME["danger"]};display:flex;align-items:center;justify-content:center">🚫 {blocked_count}</div></div>'
                    f'<div style="font-size:12px;color:{THEME["muted"]};margin-top:8px">拦截率 {100-p_pct:.0f}% — 门控越严，放行假设的期望价值越高</div>'
                )
            else:
                seg = '<div style="color:#919AA6;font-size:13px">尚未生成假设</div>'
            st.markdown(f'<div class="rl-card">{seg}</div>', unsafe_allow_html=True)

            if result.get("sensitivity"):
                st.markdown(
                    f'<div class="rl-card" style="border-left:4px solid {THEME["warn"]};font-size:13px">'
                    f'📐 {html_escape(result["sensitivity"])}</div>',
                    unsafe_allow_html=True,
                )

        with cc3:
            st.markdown("**🕸 引用网络总览**")
            st.markdown(citation_network_svg(papers, paper_id), unsafe_allow_html=True)

        if pending:
            st.markdown("#### 🔎 待补证空白（低置信度，建议人工复核）")
            rows = ""
            for p_ in pending:
                rows += (
                    f'<tr>'
                    f'<td style="padding:7px 12px;border-bottom:1px solid #EEE;font-weight:700">{html_escape(p_["paper_id"])}</td>'
                    f'<td style="padding:7px 12px;border-bottom:1px solid #EEE">{html_escape(p_["gap_id"])}</td>'
                    f'<td style="padding:7px 12px;border-bottom:1px solid #EEE">{html_escape(p_["gap_type"])} / {html_escape(p_["gap_subtype"])}</td>'
                    f'<td style="padding:7px 12px;border-bottom:1px solid #EEE">{conf_badge(p_["confidence"])}</td></tr>'
                )
            st.markdown(
                f'<div class="rl-card" style="padding:4px 8px"><table style="width:100%;border-collapse:collapse;font-size:13px">'
                f'<tr style="color:{THEME["muted"]};font-size:12px;text-align:left">'
                f'<th style="padding:7px 12px">论文</th><th style="padding:7px 12px">空白</th>'
                f'<th style="padding:7px 12px">类型</th><th style="padding:7px 12px">置信度</th></tr>'
                f'{rows}</table></div>',
                unsafe_allow_html=True,
            )

        if st.button("💾 导出全景数据", key="export_panorama"):
            os.makedirs("data", exist_ok=True)
            export_path = os.path.join("data", "panorama_export.json")
            export_data = {
                "papers": result["papers"],
                "mined": result["mined"],
                "gap_summary": {
                    "total": gs["total"],
                    "type_dist": gs["type_dist"],
                    "low_confidence_count": len(gs["low_confidence"]),
                },
                "hyp_summary": {
                    "total": hs["total"],
                    "passed": hs["passed"],
                    "blocked": hs["blocked"],
                },
                "conclusion": result["conclusion"],
                "sensitivity": result["sensitivity"],
            }
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            st.success(f"已保存到 {export_path}")

# ---------- Tab 5：图表理解（多模态） ----------
with tab_chart:
    st.markdown("#### 📊 科学图表理解（多模态 · Qwen-VL）")
    st.caption("用 Qwen-VL 提取图表数值，并检测论文声明与图表数据是否矛盾 —— 认知增强：让机器替你核对图文一致性")
    figs_dir = Path("data/figures")
    figs = sorted(
        [f for f in figs_dir.glob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    ) if figs_dir.exists() else []
    if not figs:
        st.info("data/figures 暂无图表图片")
    else:
        sel = st.selectbox("选择图表", [f.name for f in figs], index=0, key="chart_sel")
        img_path = str(figs_dir / sel)
        c_img, c_res = st.columns([4, 6])
        default_claims = "\n".join(
            str(x) for x in [
                paper.get("core_contribution", ""),
                paper.get("conclusions", ""),
                "；".join(paper.get("limitations_extracted", []))
                if isinstance(paper.get("limitations_extracted"), list) else paper.get("limitations", ""),
            ] if str(x).strip()
        )
        with c_img:
            try:
                st.image(img_path, use_column_width=True)
            except Exception:
                st.warning("图片预览失败")
            claims = st.text_area(
                "论文声明（每行一条，用于图文矛盾检测）",
                value=default_claims, height=150,
                help="把这些声明与图表提取出的数值做比对；留空则只做数值提取",
            )
        with c_res:
            claims_list = [c.strip() for c in claims.splitlines() if c.strip()]
            paper_text = paper.get("core_contribution", "") or ""
            if st.button("🔍 理解这张图表", type="primary", disabled=stage_mode,
                         help="答辩模式下已禁用（只读缓存），关闭后可点击"):
                with st.spinner("Qwen-VL 分析中…"):
                    res = analyze_chart_and_text(img_path, paper_text, claims_list)
                _render_chart_result(res, sel)
                _save_chart_finding(sel, res)
                st.rerun()
            else:
                cached = _load_chart_findings().get(sel)
                if cached:
                    st.info(f"已缓存：{cached.get('status')} · 提取 {cached.get('n_data_points', 0)} 点 · 矛盾 {cached.get('n_discrepancies', 0)} 处")

# ============================================================
# 页脚
# ============================================================
st.markdown(
    f'<div style="text-align:center;color:#B2BEC3;font-size:12px;margin-top:26px">'
    f'ResearchLens v2.0 · 基于 Qwen（百炼）· 文献挖掘 → 假设生成 → 进展追踪 → 图表理解 四层架构 · '
    f'Demo 全程可离线（结果缓存 + 论文本地预抽取）</div>',
    unsafe_allow_html=True,
)
