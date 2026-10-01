# -*- coding: utf-8 -*-
"""
图表理解引擎 v1.0

基于 Qwen-VL 多模态能力理解科学图表，提取关键数值，检测图文矛盾。
架构复用 vision.py 的缓存+降级+错误处理模式。
"""

import json
import os
import re
import hashlib
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm import call_qwen_multimodal, extract_json
from taxonomy import CHART_TYPES, DISCREPANCY_TYPES

# 缓存路径
CACHE_PATH = Path("data") / "chart_cache.json"


def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_cache() -> Dict:
    """加载图表理解缓存"""
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache: Dict):
    """保存图表理解缓存"""
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def extract_chart_data(chart_image: str, chart_type: str = None) -> Optional[Dict]:
    """
    从图表图片中提取结构化数据
    
    参数:
        chart_image: 图表图片路径
        chart_type: 图表类型（可选，如 "bar", "line", "heatmap"）
    
    返回:
        提取的数值字典，失败返回 None
    """
    if not os.path.exists(chart_image):
        print(f"  [警告] 图表文件不存在: {chart_image}")
        return None
    
    # 生成缓存 key
    with open(chart_image, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    cache_key = f"{file_hash}_{chart_type or 'auto'}"
    
    # 检查缓存
    cache = load_cache()
    if cache_key in cache:
        print(f"  [缓存命中] {chart_image}")
        return cache[cache_key]
    
    # 编码图片
    image_base64 = encode_image_to_base64(chart_image)
    
    # 构造提示词
    if chart_type and chart_type in CHART_TYPES:
        type_hint = f"这是一个{CHART_TYPES[chart_type]}。"
    else:
        type_hint = "请先识别这是什么类型的图表（柱状图/折线图/热力图/混淆矩阵/表格等）。"
    
    prompt = f"""{type_hint}

请从这张图表中提取关键数值数据，并以 JSON 格式返回：

{{
  "chart_type": "图表类型（bar/line/heatmap/confusion_matrix/table/other）",
  "title": "图表标题（如果有）",
  "axes": {{
    "x_label": "X轴标签",
    "y_label": "Y轴标签"
  }},
  "data_points": [
    {{"label": "数据点标签", "value": 数值, "unit": "单位（如果有）"}}
  ],
  "key_findings": ["从图表中可以得出的关键发现"]
}}

要求：
1. 数值必须准确，不要估算
2. 如果数值无法精确读取，标注 "approximate": true
3. 只返回 JSON，不要其他文字
"""
    
    try:
        # 调用多模态模型
        response = call_qwen_multimodal(
            image_base64=image_base64,
            prompt=prompt,
            model="qwen-vl-plus"
        )
        
        if not response:
            print(f"  [警告] Qwen-VL 返回空响应")
            return None
        
        # 解析 JSON
        result = extract_json(response)
        if not result:
            print(f"  [警告] JSON 解析失败")
            return None
        
        # 存入缓存
        cache[cache_key] = result
        save_cache(cache)
        
        print(f"  [提取成功] 从 {chart_image} 提取了 {len(result.get('data_points', []))} 个数据点")
        return result
        
    except Exception as e:
        print(f"  [错误] 图表理解失败: {e}")
        return None


def detect_discrepancy(
    chart_data: Dict,
    paper_text: str,
    paper_claims: List[str]
) -> List[Dict]:
    """
    检测图表数据与论文文本之间的矛盾
    
    参数:
        chart_data: 从图表提取的数据
        paper_text: 论文相关段落文本
        paper_claims: 论文中的声明列表（如 "准确率提升了5%"）
    
    返回:
        矛盾列表，每项包含矛盾类型、描述、证据
    """
    discrepancies = []
    
    # 提取图表中的关键数值
    chart_values = {}
    for point in chart_data.get("data_points", []):
        label = point.get("label", "").lower()
        value = point.get("value")
        if value is not None:
            chart_values[label] = value
    
    # 检查每个论文声明
    for claim in paper_claims:
        claim_lower = claim.lower()
        
        # 提取声明中的数值（跳过紧跟字母的数字，如 "F1" 中的 "1"）
        numbers = re.findall(r'(?<![a-zA-Z])(\d+\.?\d*)\s*%?', claim)
        
        if not numbers:
            continue
        
        # 尝试匹配图表数据
        for label, chart_value in chart_values.items():
            # 检查声明是否提到了这个标签
            if label in claim_lower or any(keyword in claim_lower for keyword in label.split()):
                # 提取声明中的数值
                for num_str in numbers:
                    try:
                        claimed_value = float(num_str)
                        
                        # 检查数值是否匹配（允许 10% 误差）
                        if chart_value > 0:
                            diff_ratio = abs(claimed_value - chart_value) / chart_value
                            
                            if diff_ratio > 0.1:  # 超过 10% 差异
                                discrepancy = {
                                    "type": "value_mismatch",
                                    "description": f"论文声明 '{claim}' 与图表数据不匹配",
                                    "paper_claim": claim,
                                    "claimed_value": claimed_value,
                                    "chart_value": chart_value,
                                    "difference_ratio": round(diff_ratio, 2),
                                    "evidence": {
                                        "chart_label": label,
                                        "chart_data_point": chart_data["data_points"][[p["label"].lower() for p in chart_data["data_points"]].index(label)]
                                    }
                                }
                                discrepancies.append(discrepancy)
                    except ValueError:
                        continue
    
    # 检查趋势矛盾
    if "key_findings" in chart_data:
        for finding in chart_data["key_findings"]:
            finding_lower = finding.lower()
            
            # 检查是否有反向趋势
            for claim in paper_claims:
                claim_lower = claim.lower()
                
                # 简单的关键词匹配（可以后续用 NLP 增强）
                positive_words = ["提升", "增加", "improve", "increase", "better", "higher"]
                negative_words = ["下降", "减少", "decrease", "drop", "worse", "lower"]
                
                chart_positive = any(w in finding_lower for w in positive_words)
                chart_negative = any(w in finding_lower for w in negative_words)
                claim_positive = any(w in claim_lower for w in positive_words)
                claim_negative = any(w in claim_lower for w in negative_words)
                
                if (chart_positive and claim_negative) or (chart_negative and claim_positive):
                    discrepancy = {
                        "type": "trend_contradiction",
                        "description": "图表显示的趋势与论文声明的趋势相反",
                        "chart_finding": finding,
                        "paper_claim": claim,
                        "evidence": {
                            "chart_trend": "positive" if chart_positive else "negative",
                            "claim_trend": "positive" if claim_positive else "negative"
                        }
                    }
                    discrepancies.append(discrepancy)
    
    return discrepancies


# extract_json 已统一到 llm.py（M1）：from llm import extract_json


def analyze_chart_and_text(
    chart_image: str,
    paper_text: str,
    paper_claims: List[str],
    chart_type: str = None
) -> Dict:
    """
    完整的图表-文本分析流程
    
    参数:
        chart_image: 图表图片路径
        paper_text: 论文相关段落
        paper_claims: 论文声明列表
        chart_type: 图表类型（可选）
    
    返回:
        分析结果字典
    """
    result = {
        "chart_image": chart_image,
        "chart_data": None,
        "discrepancies": [],
        "status": "success"
    }
    
    # 步骤 1: 提取图表数据
    chart_data = extract_chart_data(chart_image, chart_type)
    if not chart_data:
        result["status"] = "chart_extraction_failed"
        return result
    
    result["chart_data"] = chart_data
    
    # 步骤 2: 检测矛盾
    if paper_text and paper_claims:
        discrepancies = detect_discrepancy(chart_data, paper_text, paper_claims)
        result["discrepancies"] = discrepancies
        
        if discrepancies:
            print(f"  [发现矛盾] 检测到 {len(discrepancies)} 处图文矛盾")
            for d in discrepancies:
                print(f"    - {d['type']}: {d['description']}")
        else:
            print(f"  [一致性良好] 图表数据与论文声明一致")
    
    return result


def print_analysis_result(result: Dict):
    """打印分析结果的友好格式"""
    print("\n" + "="*60)
    print("图表理解分析报告")
    print("="*60)
    
    print(f"\n图表文件: {result['chart_image']}")
    print(f"状态: {result['status']}")
    
    if result["chart_data"]:
        cd = result["chart_data"]
        print(f"\n【图表信息】")
        print(f"  类型: {cd.get('chart_type', '未知')}")
        print(f"  标题: {cd.get('title', '无')}")
        
        if "axes" in cd:
            print(f"  X轴: {cd['axes'].get('x_label', '无')}")
            print(f"  Y轴: {cd['axes'].get('y_label', '无')}")
        
        print(f"\n【数据点】({len(cd.get('data_points', []))} 个)")
        for point in cd.get("data_points", [])[:10]:  # 最多显示 10 个
            label = point.get("label", "?")
            value = point.get("value", "?")
            unit = point.get("unit", "")
            approx = " (约)" if point.get("approximate") else ""
            print(f"  - {label}: {value}{unit}{approx}")
        
        if cd.get("key_findings"):
            print(f"\n【关键发现】")
            for finding in cd["key_findings"]:
                print(f"  • {finding}")
    
    if result["discrepancies"]:
        print(f"\n【图文矛盾】({len(result['discrepancies'])} 处)")
        for i, d in enumerate(result["discrepancies"], 1):
            print(f"\n  矛盾 {i}:")
            print(f"    类型: {d['type']}")
            print(f"    描述: {d['description']}")
            
            if d["type"] == "value_mismatch":
                print(f"    论文声明: {d['paper_claim']}")
                print(f"    论文数值: {d['claimed_value']}")
                print(f"    图表数值: {d['chart_value']}")
                print(f"    差异比例: {d['difference_ratio']*100:.1f}%")
            elif d["type"] == "trend_contradiction":
                print(f"    图表发现: {d['chart_finding']}")
                print(f"    论文声明: {d['paper_claim']}")
    else:
        print(f"\n【一致性检查】✓ 未发现明显矛盾")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # 测试模式
    print("图表理解引擎 - 测试模式\n")
    
    # 测试用例 1: 提取图表数据（如果有测试图片）
    test_image = "data/figures/test_chart.png"
    if os.path.exists(test_image):
        print(f"测试图表: {test_image}")
        result = analyze_chart_and_text(
            chart_image=test_image,
            paper_text="我们的方法在准确率上达到了 85.3%。",
            paper_claims=["准确率达到了 85.3%", "性能提升了 10%"],
            chart_type="bar"
        )
        print_analysis_result(result)
    else:
        print(f"提示: 将测试图表放到 {test_image} 即可运行测试")
        print("支持的格式: PNG, JPG, JPEG")
