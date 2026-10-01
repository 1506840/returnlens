# -*- coding: utf-8 -*-
"""
ResearchLens · 研究进展追踪 v1.0
纯本地聚合，不调用 LLM。
从挖掘结果和假设结果聚合出研究全景：假设总数/已验证/高潜力排名/待补清单/结论句。
架构源自 review.py，全面迁移到科研场景。
"""

import os
import json
from taxonomy import GAP_TYPE_LIST, GAP_SUBTYPE_MAP

# ── 路径常量 ──────────────────────────────────────────────────
SEED_PATH = os.path.join("data", "lit_seed.jsonl")
MINING_PATH = os.path.join("data", "mining_results.jsonl")
HYPOTHESIS_PATH = os.path.join("data", "hypotheses.jsonl")


def load_papers():
    """加载种子论文列表"""
    if not os.path.exists(SEED_PATH):
        return []
    papers = []
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def load_mining_results():
    """加载文献挖掘结果"""
    if not os.path.exists(MINING_PATH):
        return {}
    results = {}
    with open(MINING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                results[obj.get("paper_id")] = obj
    return results


def load_hypotheses():
    """加载假设生成结果"""
    if not os.path.exists(HYPOTHESIS_PATH):
        return {}
    results = {}
    with open(HYPOTHESIS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                results[obj.get("cache_key")] = obj.get("hypotheses", [])
    return results


def count_gaps(mining_results):
    """统计研究空白分布"""
    gap_type_dist = {}
    gap_subtype_dist = {}
    total_gaps = 0
    low_confidence_gaps = []

    for paper_id, result in mining_results.items():
        gaps = result.get("gaps", [])
        for g in gaps:
            total_gaps += 1
            gt = g.get("gap_type", "未知")
            gst = g.get("gap_subtype", "未知")
            gap_type_dist[gt] = gap_type_dist.get(gt, 0) + 1
            key = f"{gt}/{gst}"
            gap_subtype_dist[key] = gap_subtype_dist.get(key, 0) + 1

            conf = g.get("confidence", 1.0)
            if conf <= 0.5:
                low_confidence_gaps.append({
                    "paper_id": paper_id,
                    "gap_id": g.get("gap_id", "?"),
                    "gap_type": gt,
                    "gap_subtype": gst,
                    "confidence": conf,
                })

    return {
        "total": total_gaps,
        "type_dist": gap_type_dist,
        "subtype_dist": gap_subtype_dist,
        "low_confidence": low_confidence_gaps,
    }


def summarize_hypotheses(all_hypotheses):
    """汇总假设状态"""
    total = len(all_hypotheses)
    passed = 0
    blocked = 0
    passed_list = []
    blocked_list = []

    for h in all_hypotheses:
        if h.get("gate_passed"):
            passed += 1
            passed_list.append(h)
        else:
            blocked += 1
            blocked_list.append(h)

    # 按总分排序
    passed_list.sort(key=lambda x: x.get("total_score", 0), reverse=True)

    return {
        "total": total,
        "passed": passed,
        "blocked": blocked,
        "ranked": passed_list,
        "blocked_list": blocked_list,
    }


def build_conclusion(gap_summary, hyp_summary):
    """生成结论句（按 gap_type 分支给建议）"""
    total_gaps = gap_summary["total"]
    total_hyps = hyp_summary["total"]
    passed = hyp_summary["passed"]
    blocked = hyp_summary["blocked"]

    if total_gaps == 0:
        return "尚未发现研究空白，请检查文献挖掘结果。"

    if total_hyps == 0:
        return f"发现 {total_gaps} 个研究空白，但尚未生成假设。"

    # 找到最多的空白类型
    type_dist = gap_summary["type_dist"]
    if type_dist:
        top_type = max(type_dist, key=type_dist.get)
        top_count = type_dist[top_type]
    else:
        top_type = "未知"
        top_count = 0

    # 找到最高分假设
    ranked = hyp_summary["ranked"]
    if ranked:
        best = ranked[0]
        best_name = best.get("statement", "?")[:60]
        best_score = best.get("total_score", 0)
    else:
        best_name = "无"
        best_score = 0

    advice_map = {
        "方法局限": "建议优先探索方法改进方向，当前存在较多计算效率与理论保证方面的空白",
        "场景未覆盖": "建议优先关注跨域泛化和低资源场景，这些方向尚未被充分探索",
        "结论争议": "建议优先设计对照实验解决现有结论争议，特别是基线过时和复现性问题",
        "假设可放松": "建议检验现有方法的前提假设是否可放松，可能发现更通用的方法",
        "度量缺陷": "建议优先构建更完善的评估体系，当前评估指标和基准存在改进空间",
    }

    advice = advice_map.get(top_type, "建议优先推进高潜力假设的实验验证")

    low_conf = len(gap_summary["low_confidence"])
    low_conf_note = f"，其中 {low_conf} 个待补证" if low_conf > 0 else ""

    return (
        f"当前研究领域已分析 {total_gaps} 个研究空白{low_conf_note}，"
        f"生成 {total_hyps} 条假设（{passed} 条通过门控，{blocked} 条被拦截）。"
        f"「{top_type}」类空白最多（{top_count} 个）。"
        f"最高分假设：「{best_name}」（得分 {best_score:.3f}）。{advice}。"
    )


def build_sensitivity(gap_summary, hyp_summary):
    """敏感性分析（预算变化对排名的影响）"""
    ranked = hyp_summary["ranked"]
    if len(ranked) < 2:
        return ""

    # 检查当前第1和第2名
    best = ranked[0]
    second = ranked[1]
    diff = best.get("total_score", 0) - second.get("total_score", 0)

    # 检查工时分布
    efforts = [h.get("effort_days", 0) for h in ranked]
    min_effort = min(efforts) if efforts else 0
    max_effort = max(efforts) if efforts else 0

    return (
        f"当前第1名「{best.get('hyp_id', '?')}」相对第2名领先 {diff:.3f} 分。"
        f"假设工时范围为 {min_effort}-{max_effort} 人日。"
        f"若预算从 15 人日放宽到 45 人日，部分高工时高潜力假设可能进入可行范围。"
    )


def track(verbose=True):
    """主流程：聚合研究进展"""
    papers = load_papers()
    if not papers:
        if verbose:
            print("❌ 未读到种子论文")
        return None

    mining = load_mining_results()
    hypotheses = load_hypotheses()

    if not mining:
        if verbose:
            print("❌ 无挖掘结果，请先运行 literature_mining.py")
        return None

    # 收集所有假设
    all_hyps = []
    for key, hyps in hypotheses.items():
        all_hyps.extend(hyps)

    # 聚合
    gap_summary = count_gaps(mining)
    hyp_summary = summarize_hypotheses(all_hyps)
    conclusion = build_conclusion(gap_summary, hyp_summary)
    sensitivity = build_sensitivity(gap_summary, hyp_summary)

    # 待补证清单（低置信度空白）
    pending = gap_summary["low_confidence"]

    return {
        "papers": len(papers),
        "mined": len(mining),
        "gap_summary": gap_summary,
        "hyp_summary": hyp_summary,
        "pending": pending,
        "conclusion": conclusion,
        "sensitivity": sensitivity,
    }


if __name__ == "__main__":
    print("=== ResearchLens · 研究进展追踪 ===\n")

    papers = load_papers()
    print(f"✅ 读到 {len(papers)} 篇种子论文: {[p['paper_id'] for p in papers]}")

    mining = load_mining_results()
    print(f"{'✅ 找到' if mining else '⚠️ 缺少'} 挖掘结果: {len(mining)} 篇")

    hypotheses = load_hypotheses()
    all_hyps = []
    for key, hyps in hypotheses.items():
        all_hyps.extend(hyps)
    print(f"{'✅ 找到' if hypotheses else '⚠️ 缺少'} 假设: {len(all_hyps)} 条")

    result = track(verbose=False)
    if not result:
        raise SystemExit(1)

    print(f"\n论文数: {result['papers']}  已挖掘: {result['mined']}")

    gs = result["gap_summary"]
    hs = result["hyp_summary"]

    print(f"\n【研究空白】共 {gs['total']} 个")
    print(f"  类型分布: {gs['type_dist']}")
    print(f"  子类型分布: {gs['subtype_dist']}")
    print(f"  低置信度: {len(gs['low_confidence'])} 个")

    print(f"\n【假设】共 {hs['total']} 条")
    print(f"  通过门控: {hs['passed']}  被拦截: {hs['blocked']}")

    if hs["ranked"]:
        print(f"\n【高潜力假设排名】")
        for i, h in enumerate(hs["ranked"][:5], 1):
            score = h.get("total_score", 0)
            kind = h.get("hypothesis_kind", "?")
            effort = h.get("effort_days", "?")
            print(f"  #{i} [{h.get('hyp_id', '?')}] {h.get('statement', '?')[:60]}")
            print(f"       得分={score:.3f}  类型={kind}  工时={effort}天")

    if hs["blocked_list"]:
        print(f"\n【被拦截假设】{len(hs['blocked_list'])} 条")
        for h in hs["blocked_list"]:
            print(f"  🚫 [{h.get('hyp_id', '?')}] {h.get('gate_reason', '?')}")

    if result["pending"]:
        print(f"\n⚠️ 待补证空白 {len(result['pending'])} 个")
        for p in result["pending"]:
            print(f"  [{p['paper_id']}/{p['gap_id']}] {p['gap_type']}/{p['gap_subtype']}  "
                  f"置信度={p['confidence']}")

    if result["sensitivity"]:
        print(f"\n💡 {result['sensitivity']}")

    print(f"\n💡 {result['conclusion']}")
