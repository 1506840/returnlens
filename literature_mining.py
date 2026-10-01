# -*- coding: utf-8 -*-
"""
ResearchLens · 文献挖掘引擎 v1.0
输入：论文全文  →  输出：五元组 + 研究空白列表 + 置信度 + 证据引用
源自 attribution.py 架构，prompt/schema/validate/repair 全面改造为科研场景。
"""

import os
import sys
import json

from llm import call_qwen, extract_json, API_KEY
from taxonomy import GAP_TYPE_LIST, GAP_SUBTYPE_MAP

# ── 路径常量 ──────────────────────────────────────────────────
SEED_PATH = os.path.join("data", "lit_seed.jsonl")
PAPERS_DIR = os.path.join("data", "papers")
CACHE_PATH = os.path.join("data", "mining_results.jsonl")

# ── 构造枚举字符串（注入 prompt） ────────────────────────────
def _build_taxonomy_prompt():
    lines = []
    for gt, subtypes in GAP_SUBTYPE_MAP.items():
        lines.append(f"  {gt} → {'/'.join(subtypes)}")
    return "\n".join(lines)

TAXONOMY_TEXT = _build_taxonomy_prompt()

SYSTEM_PROMPT = f"""你是科研文献分析专家，只做"结构化抽取与研究空白发现"，不负责生成新假设。
仅输出一个 JSON 对象，不得有 ```json 包裹、不得有任何额外文字。

【封闭枚举 — gap_type 只能选以下值】
{TAXONOMY_TEXT}

【字段】
five_tuple: {{
  research_question: 研究解决什么问题（一句话）
  methodology: 核心方法/模型/算法（一段话）
  datasets: 使用的数据集和基准（列举）
  conclusions: 主要结论（一段话）
  limitations: 作者自述的局限性（一段话，必须基于论文原文）
}}
gaps: [
  {{
    gap_id: "G1" / "G2" / ...
    gap_type: 从封闭枚举中选择
    gap_subtype: 从对应二级枚举中选择
    evidence: [{{ quote: 论文原文摘录（必须是原文字句，不得改写或推断）, section: 所在章节名 }}]
    confidence: 0~1（对该空白真实性的置信度）
  }}
]
needs_more_evidence: bool（论文信息是否不足以充分判断空白）
ask_user: 需补充的信息，无则 null
risk_flag: 该论文的方法论风险描述，无则 null

【判定顺序】
第一步：从论文全文中抽取五元组，每个字段必须有原文支撑。
第二步：基于五元组和全文，发现研究空白。每条空白的 quote 必须是论文中的原文字句。
第三步：对每条空白评估置信度。证据不足（如仅基于推测而非论文明确陈述）→ confidence ≤ 0.5。
第四步：填 risk_flag（方法论层面的风险，不影响空白发现）。

【硬约束】
1. evidence 中的 quote 必须是论文原文的逐字摘录，不得改写、概括或推断。
2. 每篇论文至少发现 2 个研究空白。
3. gap_type 和 gap_subtype 必须严格在封闭枚举内。
4. needs_more_evidence=true 时，所有 gap 的 confidence 必须 ≤ 0.5。
5. five_tuple 的五个字段都必须填写，不得为空。
6. 信息不足以判断时，在 ask_user 中说明需要补充什么。

仅输出 JSON 对象本身。"""


# ── 全文加载 ──────────────────────────────────────────────────
def load_paper_text(paper):
    """加载论文全文。优先本地 .txt 文件。"""
    txt_path = paper.get("local_txt", "")
    if txt_path:
        full_path = os.path.join(os.path.dirname(__file__) or ".", txt_path)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
    return None


# ── validate ──────────────────────────────────────────────────
def validate(result, full_text=""):
    """校验模型输出是否符合 schema 和约束。"""
    errors = []

    if not isinstance(result, dict):
        return False, ["结果不是对象"]

    # 五元组完整性
    ft = result.get("five_tuple")
    if not isinstance(ft, dict):
        errors.append("five_tuple 缺失或不是对象")
    else:
        for field in ["research_question", "methodology", "datasets", "conclusions", "limitations"]:
            if not ft.get(field):
                errors.append(f"five_tuple.{field} 为空")

    # gaps 数组
    gaps = result.get("gaps")
    if not isinstance(gaps, list) or len(gaps) < 2:
        errors.append(f"gaps 数量不足（要求 ≥2，实际 {len(gaps) if isinstance(gaps, list) else 0}）")
    else:
        all_gap_types = set(GAP_TYPE_LIST)
        for i, g in enumerate(gaps):
            if not isinstance(g, dict):
                errors.append(f"gaps[{i}] 不是对象")
                continue
            # gap_id
            if not g.get("gap_id"):
                errors.append(f"gaps[{i}].gap_id 缺失")
            # gap_type 枚举
            gt = g.get("gap_type")
            if gt not in all_gap_types:
                errors.append(f"gaps[{i}].gap_type 非法: {gt}")
            # gap_subtype 枚举
            gst = g.get("gap_subtype")
            valid_subtypes = GAP_SUBTYPE_MAP.get(gt, [])
            if valid_subtypes and gst not in valid_subtypes:
                errors.append(f"gaps[{i}].gap_subtype 非法: {gst}（{gt} 下可选: {valid_subtypes}）")
            # confidence 范围
            conf = g.get("confidence")
            if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
                errors.append(f"gaps[{i}].confidence 非法: {conf}")
            # evidence 非空
            evs = g.get("evidence")
            if not isinstance(evs, list) or len(evs) == 0:
                errors.append(f"gaps[{i}].evidence 为空")
            elif isinstance(evs, list):
                for j, ev in enumerate(evs):
                    if not isinstance(ev, dict) or not ev.get("quote"):
                        errors.append(f"gaps[{i}].evidence[{j}].quote 缺失")

    # needs_more_evidence 联动
    if result.get("needs_more_evidence") and isinstance(gaps, list):
        for g in gaps:
            if isinstance(g, dict) and isinstance(g.get("confidence"), (int, float)):
                if g["confidence"] > 0.5:
                    errors.append(f"gaps[{g.get('gap_id','?')}].confidence={g['confidence']} > 0.5 但 needs_more_evidence=true")

    # quote 全文命中校验（软校验，失败打 warn 不阻断）
    quote_hits = 0
    quote_total = 0
    if full_text and isinstance(gaps, list):
        text_lower = " ".join(full_text.lower().split())  # normalize whitespace
        for g in gaps:
            if not isinstance(g, dict):
                continue
            for ev in (g.get("evidence") or []):
                if isinstance(ev, dict) and ev.get("quote"):
                    quote_total += 1
                    q_norm = " ".join(ev["quote"].lower().split())
                    if q_norm in text_lower:
                        quote_hits += 1

    if errors:
        return False, errors

    # 返回 quote 命中率作为附加信息
    hit_info = ""
    if quote_total > 0:
        hit_rate = quote_hits / quote_total
        hit_info = f"quote 命中率: {quote_hits}/{quote_total} ({hit_rate:.0%})"

    return True, [hit_info] if hit_info else ["OK"]


# ── repair ────────────────────────────────────────────────────
def repair(result, full_text=""):
    """自动修复常见的输出问题。"""
    fixed = []
    if not isinstance(result, dict):
        return result

    # confidence 修正：先联动降 confidence，再 clamp 到 [0,1] 范围
    gaps = result.get("gaps", [])
    if isinstance(gaps, list):
        for g in gaps:
            if not isinstance(g, dict):
                continue
            conf = g.get("confidence")
            if not isinstance(conf, (int, float)):
                continue
            # 第一步：needs_more_evidence 联动（优先于范围修正）
            if result.get("needs_more_evidence") and conf > 0.5:
                g["confidence"] = 0.5
                conf = 0.5
                fixed.append(f"{g.get('gap_id','?')} confidence 降为 0.5（needs_more_evidence=true）")
            # 第二步：越界 clamp 到 [0,1]
            if g["confidence"] < 0:
                g["confidence"] = 0.0
                fixed.append(f"{g.get('gap_id','?')} confidence 修正为 0")
            elif g["confidence"] > 1:
                g["confidence"] = 1.0
                fixed.append(f"{g.get('gap_id','?')} confidence 修正为 1")

    # gap_type/gap_subtype 非法时降级
    valid_types = set(GAP_TYPE_LIST)
    if isinstance(gaps, list):
        for g in gaps:
            if not isinstance(g, dict):
                continue
            gt = g.get("gap_type")
            if gt not in valid_types:
                g["gap_type"] = "方法局限"
                g["gap_subtype"] = "理论保证缺失"
                fixed.append(f"{g.get('gap_id','?')} gap_type 降级为默认值")
            else:
                gst = g.get("gap_subtype")
                valid_subtypes = GAP_SUBTYPE_MAP.get(gt, [])
                if valid_subtypes and gst not in valid_subtypes:
                    g["gap_subtype"] = valid_subtypes[0]
                    fixed.append(f"{g.get('gap_id','?')} gap_subtype 降级为 {valid_subtypes[0]}")

    # five_tuple 缺失字段补空串
    ft = result.get("five_tuple")
    if isinstance(ft, dict):
        for field in ["research_question", "methodology", "datasets", "conclusions", "limitations"]:
            if not ft.get(field):
                ft[field] = "（未抽取到）"
                fixed.append(f"five_tuple.{field} 补默认值")

    if fixed:
        result["_repaired"] = fixed
    return result


# ── quote 命中校验（独立函数，供外部调用） ────────────────────
def check_quote_hits(result, full_text):
    """返回 (hits, total, details) — 供答辩演示用。"""
    if not full_text or not isinstance(result, dict):
        return 0, 0, []
    text_lower = " ".join(full_text.lower().split())
    hits, total, details = 0, 0, []
    for g in (result.get("gaps") or []):
        if not isinstance(g, dict):
            continue
        for ev in (g.get("evidence") or []):
            if isinstance(ev, dict) and ev.get("quote"):
                total += 1
                q_norm = " ".join(ev["quote"].lower().split())
                hit = q_norm in text_lower
                if hit:
                    hits += 1
                details.append({
                    "gap_id": g.get("gap_id"),
                    "quote": ev["quote"][:80],
                    "hit": hit,
                })
    return hits, total, details


# ── 缓存 ──────────────────────────────────────────────────────
def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    cache = {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    cache[obj.get("paper_id", "_unknown")] = obj
    except Exception:
        return {}
    return cache


def save_result(paper_id, result):
    cache = load_cache()
    saved = {"paper_id": paper_id}
    if isinstance(result, dict):
        saved.update({k: v for k, v in result.items() if k != "paper_id"})
    else:
        saved.update({"_raw": str(result), "_error": "结果不是对象"})
    cache[paper_id] = saved
    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for v in cache.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")


# ── 主流程 ────────────────────────────────────────────────────
def mine(paper, model="qwen-plus", use_cache=True, on_progress=None):
    """对单篇论文执行文献挖掘，返回结构化结果。

    on_progress: 可选回调 fn(phase, message, data=None)，在真实里程碑点上报，
                 供界面渲染分步流水线动画（非假延时，全部对应真实计算）。
    """
    paper_id = paper["paper_id"]

    def emit(phase, message, data=None):
        if on_progress:
            on_progress(phase, message, data or {})

    if use_cache:
        cached = load_cache().get(paper_id)
        if cached:
            print(f"  [缓存命中] {paper_id}", file=sys.stderr)
            emit("cache_hit", f"本地缓存命中，直接返回（{paper_id}）",
                 {"seconds": 0})
            return cached

    # 加载全文
    full_text = load_paper_text(paper)
    if not full_text:
        err = {"_error": f"论文全文缺失: {paper.get('local_txt', '?')}"}
        save_result(paper_id, err)
        return err
    emit("load_paper", f"论文全文加载完成（{len(full_text):,} 字符，本地预抽取·零联网）")

    # 构造用户输入
    content = (
        f"论文ID: {paper_id}\n"
        f"标题: {paper['title']}\n"
        f"作者: {paper.get('authors_short', '未知')}\n"
        f"年份: {paper.get('year', '未知')}\n"
        f"核心贡献: {paper.get('core_contribution', '')}\n\n"
        f"=== 论文全文 ===\n{full_text}\n=== 全文结束 ==="
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    emit("calling_llm", f"调用 {model}：封闭枚举约束下抽取五元组并定位研究空白…")
    raw = call_qwen(messages, model=model)

    if raw is None:
        err = {"_raw": None, "_error": "模型未返回内容"}
        save_result(paper_id, err)
        return err
    emit("llm_done", f"模型返回（{len(raw):,} 字符），解析结构化输出…")

    result = extract_json(raw)
    if result is None:
        result = {"_raw": raw, "_error": "JSON 解析失败"}
    else:
        emit("validating", "schema 校验 → 枚举合法性 → 置信度联动 → 证据逐字命中检查…")
        result = repair(result, full_text)
        ok, msgs = validate(result, full_text)
        if not ok:
            result["_warn"] = msgs
        else:
            result["_info"] = msgs

    # quote 命中详情
    hits, total, details = check_quote_hits(result, full_text)
    result["_quote_check"] = {"hits": hits, "total": total, "details": details}

    result["paper_id"] = paper_id
    save_result(paper_id, result)
    emit("done", f"完成：{len(result.get('gaps', []))} 个空白，证据命中 {hits}/{total}",
         {"gaps": len(result.get("gaps", [])), "hits": hits, "total": total})
    return result


# ── 入口 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY:
        print("错误：未读到 DASHSCOPE_API_KEY，请检查 .env", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(SEED_PATH):
        print(f"错误：找不到 {SEED_PATH}", file=sys.stderr)
        sys.exit(1)

    print("=== ResearchLens · 文献挖掘引擎 ===\n", flush=True)

    papers = []
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    for paper in papers:
        pid = paper["paper_id"]
        print(f"----- {pid} | {paper['title'][:50]} | {paper.get('year', '?')} -----", flush=True)
        out = mine(paper, use_cache=False)

        if "_error" in out:
            print(f"  ❌ {out['_error']}", flush=True)
        else:
            ft = out.get("five_tuple", {})
            print(f"  研究问题: {ft.get('research_question', '?')[:80]}", flush=True)
            gaps = out.get("gaps", [])
            print(f"  发现空白: {len(gaps)} 个", flush=True)
            for g in gaps:
                print(f"    [{g.get('gap_id')}] {g.get('gap_type')}/{g.get('gap_subtype')}  "
                      f"置信度={g.get('confidence', '?')}", flush=True)
            qc = out.get("_quote_check", {})
            print(f"  quote 命中率: {qc.get('hits', 0)}/{qc.get('total', 0)}", flush=True)
            if out.get("_repaired"):
                print(f"  🔧 已自动修复: {out['_repaired']}", flush=True)
            if out.get("_warn"):
                print(f"  ⚠️ 警告: {out['_warn']}", flush=True)
        print(flush=True)

    print(f"结果已写入 {CACHE_PATH}")
