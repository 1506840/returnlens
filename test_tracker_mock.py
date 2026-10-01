# -*- coding: utf-8 -*-
"""
测试 research_tracker.py 的纯本地聚合逻辑
使用 mock 数据，不依赖 API 调用。

⚠️ 重要：mock 数据一律写入临时目录，并重定向 research_tracker 的路径常量，
   绝不允许写入真实 data/ 目录（历史教训：直接覆盖 data/lit_seed.jsonl
   曾把种子论文破坏成残缺记录）。
"""

import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(__file__))

import research_tracker


def setup_mock_data(tmpdir):
    """在临时目录创建 mock 数据文件，并把 research_tracker 的路径常量重定向过去"""
    research_tracker.SEED_PATH = os.path.join(tmpdir, "lit_seed.jsonl")
    research_tracker.MINING_PATH = os.path.join(tmpdir, "mining_results.jsonl")
    research_tracker.HYPOTHESIS_PATH = os.path.join(tmpdir, "hypotheses.jsonl")

    # 种子论文
    papers = [
        {"paper_id": "llava15", "title": "LLaVA-1.5", "year": 2023},
        {"paper_id": "som", "title": "Set-of-Mark Prompting", "year": 2023},
    ]
    with open(research_tracker.SEED_PATH, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 挖掘结果
    mining = [
        {
            "paper_id": "llava15",
            "five_tuple": {
                "research_question": "视觉指令微调",
                "methodology": "MLP连接器",
                "datasets": "VQAv2, GQA",
                "conclusions": "SOTA on 11 benchmarks",
                "limitations": "不支持多图理解"
            },
            "gaps": [
                {
                    "gap_id": "G1",
                    "gap_type": "方法局限",
                    "gap_subtype": "计算开销大",
                    "confidence": 0.85,
                    "evidence": [{"quote": "prolonged training", "section": "Limitations"}]
                },
                {
                    "gap_id": "G2",
                    "gap_type": "场景未覆盖",
                    "gap_subtype": "跨域泛化不足",
                    "confidence": 0.45,
                    "evidence": [{"quote": "limited problem solving", "section": "Limitations"}]
                }
            ]
        },
        {
            "paper_id": "som",
            "five_tuple": {
                "research_question": "视觉定位",
                "methodology": "Set-of-Mark提示",
                "datasets": "RefCOCOg, Flickr30K",
                "conclusions": "零样本超越全微调模型",
                "limitations": "开源模型无法解读标记"
            },
            "gaps": [
                {
                    "gap_id": "G1",
                    "gap_type": "场景未覆盖",
                    "gap_subtype": "跨域泛化不足",
                    "confidence": 0.75,
                    "evidence": [{"quote": "open-sourced LMMs can hardly interpret", "section": "Discussion"}]
                },
                {
                    "gap_id": "G2",
                    "gap_type": "结论争议",
                    "gap_subtype": "统计显著性不足",
                    "confidence": 0.30,
                    "evidence": [{"quote": "only a small subset", "section": "Experiments"}]
                }
            ]
        }
    ]
    with open(research_tracker.MINING_PATH, "w", encoding="utf-8") as f:
        for m in mining:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # 假设结果
    hypotheses = [
        {
            "cache_key": "llava15_budget30",
            "hypotheses": [
                {
                    "hyp_id": "H1",
                    "statement": "引入视觉注意力机制可提升长文本理解",
                    "hypothesis_kind": "改进型",
                    "gate_passed": True,
                    "gate_reason": "通过所有门控规则",
                    "total_score": 0.82,
                    "effort_days": 15,
                    "scores": {
                        "novelty": {"value": 0.9, "evidence": "新方向"},
                        "feasibility": {"value": 0.8, "evidence": "可行"},
                        "impact": {"value": 0.8, "evidence": "高影响"},
                        "evidence_strength": {"value": 0.7, "evidence": "中等证据"}
                    }
                },
                {
                    "hyp_id": "H2",
                    "statement": "多图联合推理可改善组合理解",
                    "hypothesis_kind": "统一型",
                    "gate_passed": False,
                    "gate_reason": "预估工时 45 天超出预算 30 天",
                    "total_score": 0.75,
                    "effort_days": 45,
                    "scores": {
                        "novelty": {"value": 0.8, "evidence": "新颖"},
                        "feasibility": {"value": 0.5, "evidence": "需要大量工作"},
                        "impact": {"value": 0.9, "evidence": "高影响"},
                        "evidence_strength": {"value": 0.6, "evidence": "中等"}
                    }
                }
            ]
        }
    ]
    with open(research_tracker.HYPOTHESIS_PATH, "w", encoding="utf-8") as f:
        for h in hypotheses:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")


def test_count_gaps():
    """测试：空白统计"""
    print("\n=== 测试 1: 空白统计 ===")
    mining = research_tracker.load_mining_results()
    gs = research_tracker.count_gaps(mining)

    print(f"  总空白数: {gs['total']}")
    print(f"  类型分布: {gs['type_dist']}")
    print(f"  低置信度: {len(gs['low_confidence'])}")

    assert gs["total"] == 4, f"应有4个空白，实际 {gs['total']}"
    assert gs["type_dist"]["场景未覆盖"] == 2
    assert gs["type_dist"]["方法局限"] == 1
    assert gs["type_dist"]["结论争议"] == 1
    assert len(gs["low_confidence"]) == 2, "应有2个低置信度空白（0.45和0.30）"


def test_summarize_hypotheses():
    """测试：假设汇总"""
    print("\n=== 测试 2: 假设汇总 ===")
    hypotheses = research_tracker.load_hypotheses()
    all_hyps = []
    for key, hyps in hypotheses.items():
        all_hyps.extend(hyps)

    hs = research_tracker.summarize_hypotheses(all_hyps)

    print(f"  总假设: {hs['total']}")
    print(f"  通过门控: {hs['passed']}")
    print(f"  被拦截: {hs['blocked']}")

    assert hs["total"] == 2
    assert hs["passed"] == 1
    assert hs["blocked"] == 1
    assert hs["ranked"][0]["hyp_id"] == "H1"


def test_build_conclusion():
    """测试：结论句生成"""
    print("\n=== 测试 3: 结论句生成 ===")
    mining = research_tracker.load_mining_results()
    hypotheses = research_tracker.load_hypotheses()
    all_hyps = []
    for key, hyps in hypotheses.items():
        all_hyps.extend(hyps)

    gs = research_tracker.count_gaps(mining)
    hs = research_tracker.summarize_hypotheses(all_hyps)
    conclusion = research_tracker.build_conclusion(gs, hs)

    print(f"  结论句: {conclusion}")

    assert "场景未覆盖" in conclusion, "结论应提到最多的空白类型"
    assert "H1" in conclusion or "视觉注意力" in conclusion, "结论应提到最高分假设"


def test_build_sensitivity():
    """测试：敏感性分析"""
    print("\n=== 测试 4: 敏感性分析 ===")
    hypotheses = research_tracker.load_hypotheses()
    all_hyps = []
    for key, hyps in hypotheses.items():
        all_hyps.extend(hyps)

    hs = research_tracker.summarize_hypotheses(all_hyps)
    sensitivity = research_tracker.build_sensitivity(gap_summary=None, hyp_summary=hs)

    print(f"  敏感性: {sensitivity}")

    # 只有1条通过门控，不应有敏感性分析
    assert sensitivity == "", "只有1条通过门控的假设不应有敏感性分析"


def test_track_full():
    """测试：完整追踪流程"""
    print("\n=== 测试 5: 完整追踪流程 ===")
    result = research_tracker.track(verbose=False)

    print(f"  论文数: {result['papers']}")
    print(f"  已挖掘: {result['mined']}")
    print(f"  待补证: {len(result['pending'])}")
    print(f"  结论: {result['conclusion'][:80]}...")

    assert result is not None
    assert result["papers"] == 2
    assert result["mined"] == 2
    assert len(result["pending"]) == 2
    assert result["conclusion"]


if __name__ == "__main__":
    print("=" * 60)
    print("research_tracker.py Mock 测试")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="rl_test_")
    print(f"\n创建 mock 数据（临时目录: {tmpdir}）...")
    try:
        setup_mock_data(tmpdir)
        print("✅ mock 数据已创建")

        test_count_gaps()
        test_summarize_hypotheses()
        test_build_conclusion()
        test_build_sensitivity()
        test_track_full()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！（真实 data/ 目录未被触碰）")
        print("=" * 60)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
