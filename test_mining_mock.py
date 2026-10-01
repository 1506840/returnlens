# -*- coding: utf-8 -*-
"""
测试 literature_mining.py 的 validate/repair/quote_check 逻辑
使用 mock 数据，不依赖 API 调用
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from literature_mining import validate, repair, check_quote_hits


def test_validate_pass():
    """测试：合法输入应通过校验"""
    print("\n=== 测试 1: 合法输入 ===")
    full_text = """
    LLaVA excels in conversational-style visual reasoning.
    Our model uses simple modifications to LLaVA.
    We use MLP projection and academic-task-oriented VQA data.
    Limitations include prolonged training for high-resolution images.
    """
    
    result = {
        "five_tuple": {
            "research_question": "视觉指令微调的设计选择研究",
            "methodology": "MLP 视觉-语言连接器 + 学术任务 VQA 数据",
            "datasets": "VQAv2, GQA, VizWiz, ScienceQA-IMG",
            "conclusions": "在 11 个基准上取得 SOTA",
            "limitations": "高分辨率训练时间长，不支持多图理解"
        },
        "gaps": [
            {
                "gap_id": "G1",
                "gap_type": "方法局限",
                "gap_subtype": "计算开销大",
                "evidence": [
                    {"quote": "prolonged training for high-resolution images", "section": "Limitations"}
                ],
                "confidence": 0.85
            },
            {
                "gap_id": "G2",
                "gap_type": "场景未覆盖",
                "gap_subtype": "跨域泛化不足",
                "evidence": [
                    {"quote": "conversational-style visual reasoning", "section": "Introduction"}
                ],
                "confidence": 0.75
            }
        ],
        "needs_more_evidence": False,
        "ask_user": None,
        "risk_flag": None
    }
    
    ok, msgs = validate(result, full_text)
    print(f"  校验结果: {'✅ 通过' if ok else '❌ 失败'}")
    print(f"  信息: {msgs}")
    
    hits, total, details = check_quote_hits(result, full_text)
    print(f"  Quote 命中: {hits}/{total}")
    for d in details:
        print(f"    [{d['gap_id']}] {'✅' if d['hit'] else '❌'} {d['quote']}")
    
    assert ok, f"应通过校验，但失败: {msgs}"
    assert hits == total, f"Quote 应全部命中，但 {hits}/{total}"


def test_validate_fail_missing_field():
    """测试：缺少 five_tuple 字段应失败"""
    print("\n=== 测试 2: 缺少 five_tuple 字段 ===")
    result = {
        "five_tuple": {
            "research_question": "测试问题"
            # 缺少其他字段
        },
        "gaps": [
            {
                "gap_id": "G1",
                "gap_type": "方法局限",
                "gap_subtype": "计算开销大",
                "evidence": [{"quote": "test", "section": "test"}],
                "confidence": 0.8
            },
            {
                "gap_id": "G2",
                "gap_type": "场景未覆盖",
                "gap_subtype": "跨域泛化不足",
                "evidence": [{"quote": "test2", "section": "test2"}],
                "confidence": 0.7
            }
        ],
        "needs_more_evidence": False
    }
    
    ok, msgs = validate(result)
    print(f"  校验结果: {'✅ 通过' if ok else '❌ 失败'}")
    print(f"  错误: {msgs}")
    
    assert not ok, "缺少字段应失败"
    assert any("five_tuple" in m for m in msgs), "应报告 five_tuple 字段缺失"


def test_validate_fail_invalid_gap_type():
    """测试：非法的 gap_type 应失败"""
    print("\n=== 测试 3: 非法 gap_type ===")
    result = {
        "five_tuple": {
            "research_question": "测试",
            "methodology": "测试",
            "datasets": "测试",
            "conclusions": "测试",
            "limitations": "测试"
        },
        "gaps": [
            {
                "gap_id": "G1",
                "gap_type": "非法类型",  # ← 不在枚举中
                "gap_subtype": "计算开销大",
                "evidence": [{"quote": "test", "section": "test"}],
                "confidence": 0.8
            },
            {
                "gap_id": "G2",
                "gap_type": "场景未覆盖",
                "gap_subtype": "跨域泛化不足",
                "evidence": [{"quote": "test2", "section": "test2"}],
                "confidence": 0.7
            }
        ],
        "needs_more_evidence": False
    }
    
    ok, msgs = validate(result)
    print(f"  校验结果: {'✅ 通过' if ok else '❌ 失败'}")
    print(f"  错误: {msgs}")
    
    assert not ok, "非法 gap_type 应失败"


def test_validate_fail_confidence():
    """测试：needs_more_evidence=true 但 confidence>0.5 应失败"""
    print("\n=== 测试 4: confidence 联动校验 ===")
    result = {
        "five_tuple": {
            "research_question": "测试",
            "methodology": "测试",
            "datasets": "测试",
            "conclusions": "测试",
            "limitations": "测试"
        },
        "gaps": [
            {
                "gap_id": "G1",
                "gap_type": "方法局限",
                "gap_subtype": "计算开销大",
                "evidence": [{"quote": "test", "section": "test"}],
                "confidence": 0.8  # ← 应 ≤ 0.5
            },
            {
                "gap_id": "G2",
                "gap_type": "场景未覆盖",
                "gap_subtype": "跨域泛化不足",
                "evidence": [{"quote": "test2", "section": "test2"}],
                "confidence": 0.7  # ← 应 ≤ 0.5
            }
        ],
        "needs_more_evidence": True  # ← 强制要求 confidence ≤ 0.5
    }
    
    ok, msgs = validate(result)
    print(f"  校验结果: {'✅ 通过' if ok else '❌ 失败'}")
    print(f"  错误: {msgs}")
    
    assert not ok, "confidence 联动应失败"
    assert any("confidence" in m and "0.5" in m for m in msgs), "应报告 confidence 问题"


def test_repair():
    """测试：repair 应自动修复常见问题"""
    print("\n=== 测试 5: repair 自动修复 ===")
    result = {
        "five_tuple": {
            "research_question": "测试"
            # 缺少字段，repair 应补全
        },
        "gaps": [
            {
                "gap_id": "G1",
                "gap_type": "非法类型",  # ← repair 应降级
                "gap_subtype": "非法子类型",
                "evidence": [{"quote": "test", "section": "test"}],
                "confidence": 1.5  # ← 越界，repair 应修正
            },
            {
                "gap_id": "G2",
                "gap_type": "场景未覆盖",
                "gap_subtype": "跨域泛化不足",
                "evidence": [{"quote": "test2", "section": "test2"}],
                "confidence": 0.8  # ← needs_more_evidence=true 时应降为 0.5
            }
        ],
        "needs_more_evidence": True
    }
    
    repaired = repair(result)
    print(f"  修复记录: {repaired.get('_repaired', [])}")
    
    # 检查修复结果
    assert "methodology" in repaired["five_tuple"], "应补全 five_tuple 字段"
    assert repaired["gaps"][0]["gap_type"] in ["方法局限", "场景未覆盖", "结论争议", "假设可放松", "度量缺陷"], \
        "应降级 gap_type"
    # confidence=1.5 且 needs_more_evidence=true → 先联动降为 0.5，已在 [0,1] 内不需 clamp
    assert repaired["gaps"][0]["confidence"] == 0.5, \
        f"应联动降为 0.5，实际: {repaired['gaps'][0]['confidence']}"
    assert repaired["gaps"][1]["confidence"] == 0.5, "应降为 0.5（needs_more_evidence=true）"
    
    print(f"  ✅ repair 正确修复了所有问题")


def test_quote_check():
    """测试：quote 命中检查"""
    print("\n=== 测试 6: quote 命中检查 ===")
    full_text = """
    LLaVA excels in visual reasoning.
    Our model achieves state-of-the-art results.
    Limitations include high computational cost.
    """
    
    result = {
        "gaps": [
            {
                "gap_id": "G1",
                "evidence": [
                    {"quote": "LLaVA excels in visual reasoning", "section": "Intro"}  # ✅ 命中
                ]
            },
            {
                "gap_id": "G2",
                "evidence": [
                    {"quote": "this text does not exist", "section": "Unknown"}  # ❌ 未命中
                ]
            }
        ]
    }
    
    hits, total, details = check_quote_hits(result, full_text)
    print(f"  命中: {hits}/{total}")
    for d in details:
        print(f"    [{d['gap_id']}] {'✅' if d['hit'] else '❌'} {d['quote']}")
    
    assert hits == 1 and total == 2, "应命中 1/2"


if __name__ == "__main__":
    print("=" * 60)
    print("literature_mining.py Mock 测试")
    print("=" * 60)
    
    try:
        test_validate_pass()
        test_validate_fail_missing_field()
        test_validate_fail_invalid_gap_type()
        test_validate_fail_confidence()
        test_repair()
        test_quote_check()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
