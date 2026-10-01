# -*- coding: utf-8 -*-
"""
测试 chart_understanding.py 的纯逻辑函数
使用 mock 数据，不依赖 API 调用（不调用 Qwen-VL）
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from chart_understanding import detect_discrepancy, extract_json


def test_extract_json_plain():
    """测试：纯 JSON 字符串"""
    print("\n=== 测试 1: 纯 JSON ===")
    text = '{"chart_type": "bar", "data_points": [{"label": "A", "value": 85}]}'
    result = extract_json(text)
    print(f"  解析结果: {result}")
    assert result is not None
    assert result["chart_type"] == "bar"
    assert result["data_points"][0]["value"] == 85


def test_extract_json_code_block():
    """测试：```json 包裹"""
    print("\n=== 测试 2: ```json 代码块 ===")
    text = '这是图表数据：\n```json\n{"chart_type": "line", "title": "训练曲线"}\n```\n以上是结果。'
    result = extract_json(text)
    print(f"  解析结果: {result}")
    assert result is not None
    assert result["chart_type"] == "line"
    assert result["title"] == "训练曲线"


def test_extract_json_embedded():
    """测试：JSON 嵌入在文本中"""
    print("\n=== 测试 3: JSON 嵌入文本 ===")
    text = '根据分析，结果如下：{"chart_type": "heatmap", "data_points": []} 可以看到性能提升。'
    result = extract_json(text)
    print(f"  解析结果: {result}")
    assert result is not None
    assert result["chart_type"] == "heatmap"


def test_extract_json_invalid():
    """测试：无效 JSON"""
    print("\n=== 测试 4: 无效 JSON ===")
    text = "这不是 JSON，只是一段普通文本。"
    result = extract_json(text)
    print(f"  解析结果: {result}")
    assert result is None


def test_detect_value_mismatch():
    """测试：数值不匹配矛盾检测"""
    print("\n=== 测试 5: 数值不匹配检测 ===")
    chart_data = {
        "chart_type": "bar",
        "data_points": [
            {"label": "LLaVA-1.5", "value": 36.4},
            {"label": "GPT-4V", "value": 55.7},
            {"label": "Gemini Pro", "value": 49.4}
        ],
        "key_findings": []
    }
    paper_text = "GPT-4V achieves 60.0% on the benchmark."
    paper_claims = ["GPT-4V achieves 60.0%", "LLaVA-1.5 reaches 36.4%"]
    
    discrepancies = detect_discrepancy(chart_data, paper_text, paper_claims)
    print(f"  检测到矛盾数: {len(discrepancies)}")
    for d in discrepancies:
        print(f"    [{d['type']}] {d['description']}")
    
    # GPT-4V: 论文说 60.0%，图表是 55.7%，差异 7.7% > 10%? 不，7.7/55.7=13.8% > 10%
    assert len(discrepancies) >= 1, "应检测到至少 1 个矛盾"
    assert any(d["type"] == "value_mismatch" for d in discrepancies)


def test_detect_no_discrepancy():
    """测试：一致时不应有矛盾"""
    print("\n=== 测试 6: 一致数据无矛盾 ===")
    chart_data = {
        "chart_type": "bar",
        "data_points": [
            {"label": "accuracy", "value": 85.3},
            {"label": "f1 score", "value": 82.1}
        ],
        "key_findings": []
    }
    paper_text = "The accuracy is 85.3% and F1 score is 82.1%."
    paper_claims = ["accuracy is 85.3%", "F1 score is 82.1%"]
    
    discrepancies = detect_discrepancy(chart_data, paper_text, paper_claims)
    print(f"  检测到矛盾数: {len(discrepancies)}")
    
    assert len(discrepancies) == 0, f"一致数据不应有矛盾，但检测到 {len(discrepancies)} 个"


def test_detect_trend_contradiction():
    """测试：趋势矛盾检测"""
    print("\n=== 测试 7: 趋势矛盾检测 ===")
    chart_data = {
        "chart_type": "line",
        "data_points": [],
        "key_findings": [
            "Performance decreased significantly with larger batch sizes"
        ]
    }
    paper_text = "Increasing batch size improves model performance."
    paper_claims = ["Increasing batch size improves performance"]
    
    discrepancies = detect_discrepancy(chart_data, paper_text, paper_claims)
    print(f"  检测到矛盾数: {len(discrepancies)}")
    for d in discrepancies:
        print(f"    [{d['type']}] {d['description']}")
    
    assert len(discrepancies) >= 1, "应检测到趋势矛盾"
    assert any(d["type"] == "trend_contradiction" for d in discrepancies)


def test_detect_multiple_discrepancies():
    """测试：多重矛盾同时检测"""
    print("\n=== 测试 8: 多重矛盾 ===")
    chart_data = {
        "chart_type": "bar",
        "data_points": [
            {"label": "baseline", "value": 70.0},
            {"label": "our method", "value": 72.0}
        ],
        "key_findings": [
            "Our method shows marginal improvement over baseline"
        ]
    }
    paper_text = "Our method achieves 85.0% accuracy, a significant improvement."
    paper_claims = [
        "Our method achieves 85.0%",
        "Significant improvement over baseline"
    ]
    
    discrepancies = detect_discrepancy(chart_data, paper_text, paper_claims)
    print(f"  检测到矛盾数: {len(discrepancies)}")
    for d in discrepancies:
        print(f"    [{d['type']}] {d['description']}")
    
    # 至少有数值不匹配（80.5 vs 75.2）
    assert len(discrepancies) >= 1


if __name__ == "__main__":
    print("=" * 60)
    print("chart_understanding.py Mock 测试")
    print("=" * 60)
    
    try:
        test_extract_json_plain()
        test_extract_json_code_block()
        test_extract_json_embedded()
        test_extract_json_invalid()
        test_detect_value_mismatch()
        test_detect_no_discrepancy()
        test_detect_trend_contradiction()
        test_detect_multiple_discrepancies()
        
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
