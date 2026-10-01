# -*- coding: utf-8 -*-
"""
ResearchLens · 假设生成引擎 v1.0
输入：文献挖掘结果（研究空白）  →  输出：候选假设 + 多维评分 + 门控过滤
源自 decision.py 架构，门控+评分框架完全迁移到科研场景。
"""

import json
import sys
import os
from llm import call_qwen, extract_json, API_KEY, TEXT_MODEL
from io_jsonl import read_jsonl, write_jsonl
from taxonomy import (
    HYPOTHESIS_KINDS,
    RESOURCE_TAGS,
    VERIFIABLE_VIA,
    SCORE_DIMENSIONS,
    SCORE_WEIGHTS,
    GATE_RULES,
)

CACHE_PATH = os.path.join("data", "hypotheses.jsonl")

SYSTEM_PROMPT = """你是科研假设生成专家，根据文献中发现的研究空白，生成可验证的科学假设。

【输出格式】仅输出一个 JSON 对象，不得有 ```json 包裹、不得有任何额外文字。

【字段】
hypotheses: [
  {
    hyp_id: "H1" / "H2" / ...
    gap_id: 对应的研究空白 ID
    statement: 假设陈述（一句话，必须是可验证的命题）
    hypothesis_kind: 假设类型（从封闭枚举中选择）
    rationale: 为什么这个假设有价值（一段话，必须有证据支撑）
    evidence: [原文摘录，来自论文，支撑假设的合理性]
    methodology: 验证该假设的实验方法（具体步骤）
    resource_tag: 所需资源标签（从封闭枚举中选择）
    verifiable_via: 验证方式（从封闭枚举中选择）
    effort_days: 预估所需人日（整数）
    scores: {
      novelty: { value: 0-1, evidence: "一条证据说明为什么新颖" },
      feasibility: { value: 0-1, evidence: "一条证据说明为什么可行" },
      impact: { value: 0-1, evidence: "一条证据说明影响力" },
      evidence_strength: { value: 0-1, evidence: "一条证据说明证据强度" }
    },
    gate_flags: 【门控标志】
    gate_passed: bool（系统会计算，你不需要填）,
    gate_reason: string（系统会计算，你不需要填）
  }
]

【封闭枚举】
hypothesis_kind 可选值：改进型 / 迁移型 / 反事实型 / 统一型 / 度量型
resource_tag 可选值：单机GPU / 多卡训练 / 需私有数据 / 需人工标注 / 仅需公开数据复现 / 需理论推导
verifiable_via 可选值：公开基准跑分 / 消融实验 / 统计检验 / 小规模pilot / 专家盲评

【生成原则】
1. 每个研究空白（gap）生成 1-2 个假设，不要生成过多。
2. 假设必须是可验证的命题，不能是模糊的描述。
3. rationale 和 evidence 必须基于论文原文，不能凭空编造。
4. scores 中的每个子分必须附带一条 evidence，解释为什么给这个分数。
5. effort_days 必须合理估算，基于 methodology 的具体步骤。
6. 如果某个空白不适合生成假设（比如已经被充分研究），可以生成 0 个假设，但要在 gap_id 中说明原因。

【硬约束】
1. statement 必须是可验证的命题（包含具体的变量、条件、预期结果）。
2. 每个子分（novelty/feasibility/impact/evidence_strength）必须在 0-1 之间。
3. 每个子分必须附带 evidence，不能为空。
4. hypothesis_kind / resource_tag / verifiable_via 必须在封闭枚举内。
5. gate_flags 必须如实填写：对每条门控规则，若假设确实违反/不满足则填 true，否则填 false。这是门控拦截的唯一依据，严禁一律填 false；尤其当 statement 明显反物理、无法验证或已被充分研究时，必须如实标 true。

仅输出 JSON 对象本身。"""


def _build_taxonomy_prompt():
    """构造枚举提示（注入 prompt）"""
    return f"""
假设类型：{', '.join(HYPOTHESIS_KINDS)}
资源标签：{', '.join(RESOURCE_TAGS)}
验证方式：{', '.join(VERIFIABLE_VIA)}

评分维度：
{chr(10).join([f'  - {k}: {v}' for k, v in SCORE_DIMENSIONS.items()])}
"""


def _build_gate_prompt():
    """用 taxonomy.GATE_RULES 动态构造门控标志字段说明（让 GATE_RULES 真正被使用）"""
    lines = ["{"]
    for rule, desc in GATE_RULES:
        lines.append(f'      {rule}: bool（是否{desc}，true=违反/存在问题）')
    lines.append("    }")
    return "\n".join(lines)


# 将枚举注入 prompt
SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "【封闭枚举】",
    "【封闭枚举】\n" + _build_taxonomy_prompt()
)

# 将门控规则注入 prompt（GATE_RULES 实际落地）
SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "【门控标志】",
    _build_gate_prompt()
)


def gate(hypothesis, budget_days=30):
    """门控过滤：结构校验 + 语义门控（来自 taxonomy.GATE_RULES）

    结构校验保证假设具备最低可用性（陈述/方法/证据/rationale/预算/分数范围）；
    语义门控读取 LLM 返回的 gate_flags，对 5 条规则做"违反即拦截"判定。
    任何一项失败即整体不通过，reason 汇总所有失败原因。

    返回 (passed: bool, reason: str)
    """
    fails = []

    statement = hypothesis.get("statement", "")

    # 结构规则 1: 必须有可验证的预测
    if not statement or len(statement) < 10:
        fails.append("假设陈述过短，无法构成可验证的命题")

    # 结构规则 2: 必须有实验方法
    methodology = hypothesis.get("methodology", "")
    if not methodology or len(methodology) < 20:
        fails.append("缺少具体的实验验证方法")

    # 结构规则 3: 必须有证据支撑
    evidence = hypothesis.get("evidence", [])
    if not evidence or len(evidence) == 0:
        fails.append("缺少支撑假设的文献证据")

    # 结构规则 4: 必须有 rationale
    rationale = hypothesis.get("rationale", "")
    if not rationale or len(rationale) < 20:
        fails.append("缺少假设的价值说明（rationale）")

    # 结构规则 5: effort_days 必须在预算内
    effort_days = hypothesis.get("effort_days", 999)
    if not isinstance(effort_days, (int, float)):
        fails.append(f"effort_days 类型错误: {type(effort_days)}")
    elif effort_days > budget_days:
        fails.append(f"预估工时 {effort_days} 天超出预算 {budget_days} 天")

    # 结构规则 6: 所有子分必须在 [0, 1] 范围内
    scores = hypothesis.get("scores", {})
    for dim in SCORE_DIMENSIONS:
        score_obj = scores.get(dim, {})
        if not isinstance(score_obj, dict):
            fails.append(f"评分维度 {dim} 格式错误")
            continue
        value = score_obj.get("value")
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            fails.append(f"评分维度 {dim} 的值 {value} 不在 [0, 1] 范围内")

    # 语义门控：聚合 LLM 返回的 GATE_RULES 标志
    flags = hypothesis.get("gate_flags") or {}
    for rule, desc in GATE_RULES:
        if flags.get(rule) is True:
            fails.append(f"门控拦截（{rule}）：{desc}")

    if fails:
        return False, "；".join(fails)
    return True, "通过所有门控规则（结构校验 + 语义门控）"


def score(hypothesis):
    """计算加权总分
    
    公式：total = sum(weight_i * score_i)
    权重来自 taxonomy.SCORE_WEIGHTS
    """
    scores = hypothesis.get("scores", {})
    total = 0.0
    
    for dim, weight in SCORE_WEIGHTS.items():
        score_obj = scores.get(dim, {})
        if isinstance(score_obj, dict):
            value = score_obj.get("value", 0)
            if isinstance(value, (int, float)):
                total += weight * value
    
    return round(total, 3)


def validate(hypothesis):
    """校验假设的 schema 完整性"""
    errors = []
    
    required_fields = [
        "hyp_id", "gap_id", "statement", "hypothesis_kind",
        "rationale", "evidence", "methodology", "resource_tag",
        "verifiable_via", "effort_days", "scores"
    ]
    
    for field in required_fields:
        if field not in hypothesis:
            errors.append(f"缺少必需字段: {field}")
    
    if errors:
        return False, errors
    
    # 校验枚举值
    if hypothesis["hypothesis_kind"] not in HYPOTHESIS_KINDS:
        errors.append(f"hypothesis_kind 非法: {hypothesis['hypothesis_kind']}")
    
    if hypothesis["resource_tag"] not in RESOURCE_TAGS:
        errors.append(f"resource_tag 非法: {hypothesis['resource_tag']}")
    
    if hypothesis["verifiable_via"] not in VERIFIABLE_VIA:
        errors.append(f"verifiable_via 非法: {hypothesis['verifiable_via']}")
    
    # 校验 scores 结构
    scores = hypothesis.get("scores", {})
    for dim in SCORE_DIMENSIONS:
        if dim not in scores:
            errors.append(f"缺少评分维度: {dim}")
            continue
        
        score_obj = scores[dim]
        if not isinstance(score_obj, dict):
            errors.append(f"评分维度 {dim} 不是对象")
            continue
        
        if "value" not in score_obj:
            errors.append(f"评分维度 {dim} 缺少 value")
        if "evidence" not in score_obj:
            errors.append(f"评分维度 {dim} 缺少 evidence")
    
    if errors:
        return False, errors
    
    return True, []


def repair(hypothesis):
    """自动修复常见问题"""
    fixed = []
    
    # 修复缺失字段
    if "hypothesis_kind" not in hypothesis:
        hypothesis["hypothesis_kind"] = "改进型"
        fixed.append("hypothesis_kind 默认为 '改进型'")
    
    if "resource_tag" not in hypothesis:
        hypothesis["resource_tag"] = "单机GPU"
        fixed.append("resource_tag 默认为 '单机GPU'")
    
    if "verifiable_via" not in hypothesis:
        hypothesis["verifiable_via"] = "公开基准跑分"
        fixed.append("verifiable_via 默认为 '公开基准跑分'")
    
    if "effort_days" not in hypothesis:
        hypothesis["effort_days"] = 30
        fixed.append("effort_days 默认为 30")
    
    # 修复非法枚举值
    if hypothesis.get("hypothesis_kind") not in HYPOTHESIS_KINDS:
        hypothesis["hypothesis_kind"] = "改进型"
        fixed.append(f"hypothesis_kind 非法，降级为 '改进型'")
    
    if hypothesis.get("resource_tag") not in RESOURCE_TAGS:
        hypothesis["resource_tag"] = "单机GPU"
        fixed.append(f"resource_tag 非法，降级为 '单机GPU'")
    
    if hypothesis.get("verifiable_via") not in VERIFIABLE_VIA:
        hypothesis["verifiable_via"] = "公开基准跑分"
        fixed.append(f"verifiable_via 非法，降级为 '公开基准跑分'")
    
    # 修复 scores 结构
    if "scores" not in hypothesis:
        hypothesis["scores"] = {}
        fixed.append("scores 默认为空对象")
    
    for dim in SCORE_DIMENSIONS:
        if dim not in hypothesis["scores"]:
            hypothesis["scores"][dim] = {"value": 0.5, "evidence": "自动填充"}
            fixed.append(f"scores.{dim} 默认为 0.5")
        else:
            score_obj = hypothesis["scores"][dim]
            if not isinstance(score_obj, dict):
                hypothesis["scores"][dim] = {"value": 0.5, "evidence": "自动填充"}
                fixed.append(f"scores.{dim} 格式错误，重置为默认值")
            else:
                if "value" not in score_obj:
                    score_obj["value"] = 0.5
                    fixed.append(f"scores.{dim}.value 默认为 0.5")
                if "evidence" not in score_obj:
                    score_obj["evidence"] = "自动填充"
                    fixed.append(f"scores.{dim}.evidence 默认为 '自动填充'")
                
                # 修正越界值
                value = score_obj.get("value", 0.5)
                if not isinstance(value, (int, float)):
                    score_obj["value"] = 0.5
                    fixed.append(f"scores.{dim}.value 类型错误，修正为 0.5")
                elif value < 0:
                    score_obj["value"] = 0.0
                    fixed.append(f"scores.{dim}.value 修正为 0")
                elif value > 1:
                    score_obj["value"] = 1.0
                    fixed.append(f"scores.{dim}.value 修正为 1")
    
    if fixed:
        hypothesis["_repaired"] = fixed
    
    return hypothesis


def generate(mining_result, budget_days=30, model=TEXT_MODEL, use_cache=True, on_progress=None):
    """基于文献挖掘结果生成假设
    
    Args:
        mining_result: 文献挖掘结果（来自 literature_mining.py）
        budget_days: 工时预算（天）
        model: Qwen 模型名称
        use_cache: 是否使用缓存
        on_progress: 可选回调 fn(phase, message, data=None)，真实里程碑上报
    
    Returns:
        假设列表，每个假设包含评分和门控结果
    """
    paper_id = mining_result.get("paper_id", "unknown")
    cache_key = f"{paper_id}_budget{budget_days}"

    def emit(phase, message, data=None):
        if on_progress:
            on_progress(phase, message, data or {})

    # 检查缓存
    if use_cache:
        cached = load_cache().get(cache_key)
        if cached:
            print(f"  [缓存命中] {cache_key}", file=sys.stderr)
            emit("cache_hit", f"本地缓存命中（{cache_key}）", {"count": len(cached)})
            return cached
    
    # 提取研究空白
    gaps = mining_result.get("gaps", [])
    if not gaps:
        print(f"  [警告] {paper_id} 没有研究空白", file=sys.stderr)
        return []
    emit("load_gaps", f"载入 {len(gaps)} 个研究空白作为生成依据", {"gaps": len(gaps)})
    
    # 构造 prompt
    gaps_text = []
    for i, gap in enumerate(gaps, 1):
        gap_id = gap.get("gap_id", f"G{i}")
        gap_type = gap.get("gap_type", "未知")
        gap_subtype = gap.get("gap_subtype", "未知")
        description = gap.get("description", "")
        confidence = gap.get("confidence", "?")
        evidence = gap.get("evidence", [])
        
        # evidence 是 [{quote, section, hit}] 对象列表，展开为文本
        ev_lines = []
        for ev in evidence[:3]:
            if isinstance(ev, dict):
                q = ev.get("quote", "")
                sec = ev.get("section", "")
                ev_lines.append(f'  - 「{q}」（{sec}）')
            else:
                ev_lines.append(f"  - {ev}")
        evidence_str = "\n".join(ev_lines) if ev_lines else "  - （无摘录证据）"
        desc_line = f"  描述: {description}\n" if description else ""
        gaps_text.append(f"""
空白 {gap_id}:
  类型: {gap_type} / {gap_subtype}
  置信度: {confidence}
{desc_line}  原文证据:
{evidence_str}
""")
    
    ft = mining_result.get("five_tuple", {})
    user_prompt = f"""
论文研究问题: {ft.get('research_question', '未知')}
论文方法: {ft.get('methodology', '未知')}
论文局限性: {ft.get('limitations', '未知')}

以下是从该论文中发现的研究空白：

{"".join(gaps_text)}

请基于这些研究空白，生成可验证的科学假设。每个空白生成 1-2 个假设。
"""
    
    # 调用 LLM
    emit("calling_llm", f"调用 {model}：归纳+演绎生成候选假设并四维打分…")
    raw = call_qwen([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ], model=model)
    if not raw:
        emit("failed", "LLM 调用失败")
        return {"error": "LLM 调用失败"}
    
    # 解析 JSON
    result = extract_json(raw)
    if not result or "hypotheses" not in result:
        return {"error": "JSON 解析失败", "raw": raw}
    
    # 后处理：repair → validate → gate → score
    hypotheses = result.get("hypotheses", [])
    emit("parsing", f"模型返回 {len(hypotheses)} 条候选，开始门控判定…")
    processed = []
    for idx, hyp in enumerate(hypotheses, 1):
        # 修复
        hyp = repair(hyp)
        
        # 校验
        valid, errors = validate(hyp)
        if not valid:
            hyp["_validation_errors"] = errors
            emit("gating", f"[{idx}/{len(hypotheses)}] {hyp.get('hyp_id', '?')} 校验未通过，丢弃",
                 {"decision": "dropped"})
            continue
        
        # 门控
        passed, reason = gate(hyp, budget_days)
        hyp["gate_passed"] = passed
        hyp["gate_reason"] = reason
        
        # 评分
        hyp["total_score"] = score(hyp)
        
        if passed:
            emit("gating", f"[{idx}/{len(hypotheses)}] {hyp.get('statement', '')[:36]}… "
                           f"✅ 放行（{hyp['total_score']:.3f}）",
                 {"decision": "pass", "score": hyp["total_score"]})
        else:
            emit("gating", f"[{idx}/{len(hypotheses)}] {hyp.get('statement', '')[:36]}… "
                           f"🚫 拦截：{reason}",
                 {"decision": "block", "reason": reason})
        processed.append(hyp)
    
    # 缓存
    save_cache(cache_key, processed)
    n_pass = sum(1 for h in processed if h.get("gate_passed"))
    emit("done", f"完成：{len(processed)} 条假设（放行 {n_pass} / 拦截 {len(processed) - n_pass}）",
         {"count": len(processed), "passed": n_pass})
    
    return processed


def load_cache():
    """加载缓存（返回 {cache_key: 假设列表}）"""
    raw = read_jsonl(CACHE_PATH, key="cache_key")
    return {k: obj.get("hypotheses", []) for k, obj in raw.items()}


def save_cache(cache_key, hypotheses):
    """保存缓存（统一走 io_jsonl，M-B）"""
    cache = load_cache()
    cache[cache_key] = hypotheses
    out = [{"cache_key": k, "hypotheses": hyps} for k, hyps in cache.items()]
    write_jsonl(CACHE_PATH, out)


def print_hypothesis(hyp):
    """打印单个假设的详细信息"""
    print(f"\n{'='*60}")
    print(f"假设 ID: {hyp.get('hyp_id', '?')}")
    print(f"对应空白: {hyp.get('gap_id', '?')}")
    print(f"{'='*60}")
    
    print(f"\n【假设陈述】")
    print(f"  {hyp.get('statement', '?')}")
    
    print(f"\n【假设类型】")
    print(f"  {hyp.get('hypothesis_kind', '?')}")
    
    print(f"\n【价值说明】")
    print(f"  {hyp.get('rationale', '?')}")
    
    print(f"\n【支撑证据】")
    for e in hyp.get("evidence", []):
        print(f"  - {e}")
    
    print(f"\n【实验方法】")
    print(f"  {hyp.get('methodology', '?')}")
    
    print(f"\n【资源需求】")
    print(f"  {hyp.get('resource_tag', '?')}")
    print(f"  预估工时: {hyp.get('effort_days', '?')} 天")
    
    print(f"\n【验证方式】")
    print(f"  {hyp.get('verifiable_via', '?')}")
    
    print(f"\n【评分详情】")
    scores = hyp.get("scores", {})
    for dim in ["novelty", "feasibility", "impact", "evidence_strength"]:
        score_obj = scores.get(dim, {})
        value = score_obj.get("value", 0)
        evidence = score_obj.get("evidence", "")
        weight = SCORE_WEIGHTS.get(dim, 0)
        print(f"  {dim:20s}: {value:.2f} (权重 {weight:.2f})")
        print(f"    证据: {evidence}")
    
    print(f"\n【加权总分】")
    print(f"  {hyp.get('total_score', 0):.3f}")
    
    print(f"\n【门控结果】")
    passed = hyp.get("gate_passed", False)
    reason = hyp.get("gate_reason", "?")
    status = "✅ 通过" if passed else "❌ 未通过"
    print(f"  {status}: {reason}")
    
    if hyp.get("_repaired"):
        print(f"\n【自动修复】")
        for fix in hyp["_repaired"]:
            print(f"  - {fix}")
    
    if hyp.get("_validation_errors"):
        print(f"\n【校验错误】")
        for err in hyp["_validation_errors"]:
            print(f"  - {err}")


def main():
    """主函数：从缓存读取挖掘结果，生成假设"""
    if not API_KEY:
        print("错误：未找到 API_KEY", file=sys.stderr)
        sys.exit(1)
    
    # 读取挖掘结果
    mining_cache = os.path.join("data", "mining_results.jsonl")
    if not os.path.exists(mining_cache):
        print(f"错误：未找到 {mining_cache}，请先运行 literature_mining.py", file=sys.stderr)
        sys.exit(1)
    
    mining_results = []
    with open(mining_cache, "r", encoding="utf-8") as f:
        for line in f:
            mining_results.append(json.loads(line))
    
    print(f"=== 假设生成引擎 ===")
    print(f"读取 {len(mining_results)} 篇论文的挖掘结果")
    print(f"工时预算: 30 天")
    
    all_hypotheses = []
    
    for mining in mining_results:
        paper_id = mining.get("paper_id", "?")
        print(f"\n{'='*60}")
        print(f"论文: {paper_id}")
        print(f"{'='*60}")
        
        hypotheses = generate(mining, budget_days=30, use_cache=False)
        
        if isinstance(hypotheses, dict) and "error" in hypotheses:
            print(f"  ❌ 错误: {hypotheses['error']}")
            continue
        
        print(f"  生成 {len(hypotheses)} 个假设")
        
        for hyp in hypotheses:
            print_hypothesis(hyp)
            all_hypotheses.append(hyp)
    
    # 统计
    print(f"\n{'='*60}")
    print(f"总计生成 {len(all_hypotheses)} 个假设")
    
    passed = sum(1 for h in all_hypotheses if h.get("gate_passed"))
    blocked = len(all_hypotheses) - passed
    
    print(f"通过门控: {passed}")
    print(f"被拦截: {blocked}")
    
    if all_hypotheses:
        scores = [h.get("total_score", 0) for h in all_hypotheses if h.get("gate_passed")]
        if scores:
            print(f"通过门控的假设平均得分: {sum(scores)/len(scores):.3f}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
