# -*- coding: utf-8 -*-
"""
测试 hypothesis_generator.py 的 gate/score/validate/repair 逻辑
使用 mock 数据，不依赖 API 调用
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from hypothesis_generator import gate, score, validate, repair
from taxonomy import SCORE_WEIGHTS


def make_valid_hyp():
    """构造一个合法的假设对象"""
    return {
        "hyp_id": "H1",
        "gap_id": "G1",
        "statement": "引入视觉注意力机制可以提升多模态模型在长文本理解任务上的准确率",
        "hypothesis_kind": "改进型",
        "rationale": "现有方法在处理长文本时忽略了视觉信息的细粒度对齐，引入注意力机制可以解决这个问题",
        "evidence": ["论文指出模型在长文本任务上表现不佳", "注意力机制在NLP任务中已被证明有效"],
        "methodology": "1. 在LLaVA-1.5基础上添加交叉注意力层 2. 使用长文本VQA数据集训练 3. 在MMMU-Pro基准上评估",
        "resource_tag": "单机GPU",
        "verifiable_via": "公开基准跑分",
        "effort_days": 15,
        "scores": {
            "novelty": {"value": 0.8, "evidence": "尚未有工作将交叉注意力应用于多模态长文本"},
            "feasibility": {"value": 0.9, "evidence": "基于开源模型，单机GPU即可完成训练"},
            "impact": {"value": 0.7, "evidence": "长文本理解是当前研究热点"},
            "evidence_strength": {"value": 0.6, "evidence": "论文提供了初步证据但不够充分"}
        }
    }


def test_gate_pass():
    """测试：合法假设应通过门控"""
    print("\n=== 测试 1: 合法假设通过门控 ===")
    hyp = make_valid_hyp()
    passed, reason = gate(hyp, budget_days=30)
    print(f"  门控结果: {'✅ 通过' if passed else '❌ 未通过'}")
    print(f"  原因: {reason}")
    assert passed, f"应通过门控，但失败: {reason}"


def test_gate_block_effort():
    """测试：工时超预算应被拦截"""
    print("\n=== 测试 2: 工时超预算 ===")
    hyp = make_valid_hyp()
    hyp["effort_days"] = 50
    passed, reason = gate(hyp, budget_days=30)
    print(f"  门控结果: {'✅ 通过' if passed else '❌ 未通过'}")
    print(f"  原因: {reason}")
    assert not passed, "工时超预算应被拦截"
    assert "预算" in reason


def test_gate_block_no_methodology():
    """测试：缺少实验方法应被拦截"""
    print("\n=== 测试 3: 缺少实验方法 ===")
    hyp = make_valid_hyp()
    hyp["methodology"] = ""
    passed, reason = gate(hyp, budget_days=30)
    print(f"  门控结果: {'✅ 通过' if passed else '❌ 未通过'}")
    print(f"  原因: {reason}")
    assert not passed, "缺少实验方法应被拦截"


def test_gate_block_no_evidence():
    """测试：缺少证据应被拦截"""
    print("\n=== 测试 4: 缺少证据 ===")
    hyp = make_valid_hyp()
    hyp["evidence"] = []
    passed, reason = gate(hyp, budget_days=30)
    print(f"  门控结果: {'✅ 通过' if passed else '❌ 未通过'}")
    print(f"  原因: {reason}")
    assert not passed, "缺少证据应被拦截"


def test_gate_block_bad_score():
    """测试：评分越界应被拦截"""
    print("\n=== 测试 5: 评分越界 ===")
    hyp = make_valid_hyp()
    hyp["scores"]["novelty"]["value"] = 1.5
    passed, reason = gate(hyp, budget_days=30)
    print(f"  门控结果: {'✅ 通过' if passed else '❌ 未通过'}")
    print(f"  原因: {reason}")
    assert not passed, "评分越界应被拦截"


def test_score_calculation():
    """测试：加权评分计算"""
    print("\n=== 测试 6: 加权评分计算 ===")
    hyp = make_valid_hyp()
    total = score(hyp)
    
    # 手动计算预期值
    expected = (
        SCORE_WEIGHTS["novelty"] * 0.8 +
        SCORE_WEIGHTS["feasibility"] * 0.9 +
        SCORE_WEIGHTS["impact"] * 0.7 +
        SCORE_WEIGHTS["evidence_strength"] * 0.6
    )
    expected = round(expected, 3)
    
    print(f"  计算得分: {total}")
    print(f"  预期得分: {expected}")
    assert abs(total - expected) < 0.001, f"得分计算错误: {total} != {expected}"


def test_validate_pass():
    """测试：合法假设通过校验"""
    print("\n=== 测试 7: 合法假设通过校验 ===")
    hyp = make_valid_hyp()
    valid, errors = validate(hyp)
    print(f"  校验结果: {'✅ 通过' if valid else '❌ 失败'}")
    if errors:
        print(f"  错误: {errors}")
    assert valid, f"应通过校验，但失败: {errors}"


def test_validate_fail_missing_field():
    """测试：缺少字段应失败"""
    print("\n=== 测试 8: 缺少字段 ===")
    hyp = make_valid_hyp()
    del hyp["statement"]
    valid, errors = validate(hyp)
    print(f"  校验结果: {'✅ 通过' if valid else '❌ 失败'}")
    print(f"  错误: {errors}")
    assert not valid, "缺少字段应失败"
    assert any("statement" in e for e in errors)


def test_validate_fail_bad_enum():
    """测试：非法枚举值应失败"""
    print("\n=== 测试 9: 非法枚举值 ===")
    hyp = make_valid_hyp()
    hyp["hypothesis_kind"] = "非法类型"
    valid, errors = validate(hyp)
    print(f"  校验结果: {'✅ 通过' if valid else '❌ 失败'}")
    print(f"  错误: {errors}")
    assert not valid, "非法枚举值应失败"


def test_repair():
    """测试：repair 应自动修复常见问题"""
    print("\n=== 测试 10: repair 自动修复 ===")
    hyp = {
        "hyp_id": "H1",
        "gap_id": "G1",
        "statement": "测试假设",
        "rationale": "这是一个测试假设的价值说明",
        "evidence": ["证据1"],
        "methodology": "具体的实验验证方法描述，至少二十个字",
        "effort_days": 10,
        # 缺少 hypothesis_kind, resource_tag, verifiable_via, scores
        "scores": {
            "novelty": {"value": 0.8},  # 缺少 evidence
            "feasibility": {"value": 1.5},  # 越界 + 缺少 evidence
        }
        # 缺少 impact, evidence_strength
    }
    
    repaired = repair(hyp)
    print(f"  修复记录: {repaired.get('_repaired', [])}")
    
    # 检查修复结果
    assert repaired["hypothesis_kind"] in ["改进型", "迁移型", "反事实型", "统一型", "度量型"]
    assert repaired["resource_tag"] in ["单机GPU", "多卡训练", "需私有数据", "需人工标注", "仅需公开数据复现", "需理论推导"]
    assert repaired["verifiable_via"] in ["公开基准跑分", "消融实验", "统计检验", "小规模pilot", "专家盲评"]
    assert "impact" in repaired["scores"]
    assert "evidence_strength" in repaired["scores"]
    assert "evidence" in repaired["scores"]["novelty"]
    assert "evidence" in repaired["scores"]["feasibility"]
    assert repaired["scores"]["feasibility"]["value"] == 1.0, "越界值应修正为 1.0"
    
    print(f"  ✅ repair 正确修复了所有问题")


def test_repair_bad_enums():
    """测试：repair 应修正非法枚举值"""
    print("\n=== 测试 11: repair 修正非法枚举 ===")
    hyp = make_valid_hyp()
    hyp["hypothesis_kind"] = "非法类型"
    hyp["resource_tag"] = "非法资源"
    hyp["verifiable_via"] = "非法验证"
    
    repaired = repair(hyp)
    print(f"  修复记录: {repaired.get('_repaired', [])}")
    
    assert repaired["hypothesis_kind"] == "改进型"
    assert repaired["resource_tag"] == "单机GPU"
    assert repaired["verifiable_via"] == "公开基准跑分"
    print(f"  ✅ repair 正确降级了非法枚举值")


if __name__ == "__main__":
    print("=" * 60)
    print("hypothesis_generator.py Mock 测试")
    print("=" * 60)
    
    try:
        test_gate_pass()
        test_gate_block_effort()
        test_gate_block_no_methodology()
        test_gate_block_no_evidence()
        test_gate_block_bad_score()
        test_score_calculation()
        test_validate_pass()
        test_validate_fail_missing_field()
        test_validate_fail_bad_enum()
        test_repair()
        test_repair_bad_enums()
        
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
