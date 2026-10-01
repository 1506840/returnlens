# -*- coding: utf-8 -*-
"""
图表理解引擎 v1.0

基于 Qwen-VL 多模态能力理解科学图表，提取关键数值，检测图文矛盾。
架构复用 vision.py 的缓存+降级+错误处理模式。
"""

import json
import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm import call_qwen_multimodal, extract_json, VL_MODEL
from taxonomy import CHART_TYPES

# 缓存策略（H-B 统一）：图表理解结论的唯一缓存由 UI 层写入 data/chart_findings.json
# （app.py 的 _save_chart_finding / research_tracker.load_chart_findings 共用），
# 引擎保持无状态，不再维护第二份 chart_cache.json，避免双缓存冗余。


def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
            model=VL_MODEL
        )
        
        if not response:
            print(f"  [警告] Qwen-VL 返回空响应")
            return None
        
        # 解析 JSON
        result = extract_json(response)
        if not result:
            print(f"  [警告] JSON 解析失败")
            return None
        
        # 解析成功（结论由 UI 层缓存到 data/chart_findings.json，引擎无状态）
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

    # ── 补全 taxonomy.DISCREPANCY_TYPES 声明的其余 3 类（H-B）──
    _detect_label_mismatch(chart_data, paper_claims, discrepancies)
    _detect_missing_data(chart_data, paper_claims, discrepancies)
    _detect_scale_mismatch(chart_data, paper_claims, discrepancies)

    return discrepancies


# ── 以下 3 个检测器补全 taxonomy.DISCREPANCY_TYPES 的剩余类型（H-B）──
# 均为保守启发式：只在论文有显式措辞/数值时才上报，最大限度避免误报。
_AXIS_PAT = re.compile(
    r"(横轴|纵轴|x\s*轴|y\s*轴|图例|坐标轴)\s*(表示|是|为|:|：)?\s*([一-龥A-Za-z0-9_%/．.\d]+)"
)
_MULT_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*(倍|×|x|fold|倍于)", re.IGNORECASE)
_METRIC_KW = ["准确率", "精度", "召回率", "recall", "precision", "f1", "ap",
              "损失", "loss", "auc", "bleu", "误差", "error", "mae", "rmse", "acc"]


def _detect_label_mismatch(chart_data, claims, out):
    """论文显式指称某个轴/标签，但图表轴标签与数据标签中都没有该名称。"""
    chart_labels = set()
    axes = chart_data.get("axes") or {}
    for lab in (axes.get("x_label"), axes.get("y_label")):
        if lab:
            chart_labels.add(str(lab).strip().lower())
    for p in chart_data.get("data_points", []):
        if p.get("label"):
            chart_labels.add(str(p["label"]).strip().lower())
    for claim in claims:
        for m in _AXIS_PAT.finditer(claim):
            mentioned = m.group(3).strip().lower()
            if mentioned and mentioned not in chart_labels:
                out.append({
                    "type": "label_mismatch",
                    "description": f"论文指称『{mentioned}』，但图表轴标签/数据标签中未出现该名称",
                    "paper_claim": claim,
                    "mentioned_label": mentioned,
                    "chart_labels": sorted(chart_labels),
                })


def _detect_missing_data(chart_data, claims, out):
    """论文引用了某指标的具体数值，但图表中既无对应标签也无该数值（数据缺失）。"""
    labels_lower = [str(p.get("label", "")).lower() for p in chart_data.get("data_points", [])]
    values = [p.get("value") for p in chart_data.get("data_points", [])
              if isinstance(p.get("value"), (int, float))]
    for claim in claims:
        cl = claim.lower()
        hit_kw = next((k for k in _METRIC_KW if k in cl), None)
        if not hit_kw:
            continue
        nums = re.findall(r"(?<![a-zA-Z])(\d+\.?\d*)", claim)
        claimed_vals = [float(n) for n in nums]
        label_present = any(hit_kw in lab for lab in labels_lower)
        val_present = bool(values and claimed_vals) and any(
            abs(cv - v) / max(abs(cv), 1e-9) < 0.05
            for cv in values for v in claimed_vals
        )
        if not label_present and not val_present:
            out.append({
                "type": "missing_data",
                "description": f"论文引用了『{hit_kw}』相关数据，但图表中既无对应标签也无该数值",
                "paper_claim": claim,
                "metric": hit_kw,
            })


def _detect_scale_mismatch(chart_data, claims, out):
    """论文声称的倍数变化与图表实际数据比例不符。"""
    vals = [p.get("value") for p in chart_data.get("data_points", [])
            if isinstance(p.get("value"), (int, float)) and p.get("value", 0) > 0]
    if len(vals) < 2:
        return
    chart_ratio = max(vals) / min(vals)
    for claim in claims:
        for m in _MULT_PAT.finditer(claim):
            try:
                claimed_ratio = float(m.group(1))
            except ValueError:
                continue
            if claimed_ratio <= 0:
                continue
            diff = abs(claimed_ratio - chart_ratio) / max(claimed_ratio, chart_ratio)
            if diff > 0.5:
                out.append({
                    "type": "scale_mismatch",
                    "description": f"论文声称变化约 {claimed_ratio} 倍，但图表数据比例约 {chart_ratio:.1f} 倍",
                    "paper_claim": claim,
                    "claimed_ratio": claimed_ratio,
                    "chart_ratio": round(chart_ratio, 2),
                })


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
