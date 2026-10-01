# -*- coding: utf-8 -*-
"""
ResearchLens · LLM 调用与 JSON 解析公共层
从 attribution.py:60-96 抽出，全项目共用。
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from dashscope import Generation

load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")


def call_qwen(messages, model="qwen-plus", max_retries=2):
    """调用 Qwen API，返回原始文本；失败返回 None。"""
    for attempt in range(max_retries + 1):
        try:
            resp = Generation.call(
                model=model,
                messages=messages,
                api_key=API_KEY,
                result_format="message",
                temperature=0.1,
            )
            if resp.status_code == 200:
                return resp.output.choices[0].message.content
            print(f"  [重试 {attempt + 1}] HTTP {resp.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"  [重试 {attempt + 1}] {e}", file=sys.stderr)
        time.sleep(1.5)
    return None


def extract_json(text):
    """从模型输出中提取 JSON 对象；失败返回 None。
    兼容 ```json 包裹、前后多余文字等情况。"""
    if not text:
        return None
    text = text.strip()
    # 去掉 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    lo, hi = text.find("{"), text.rfind("}")
    if lo == -1 or hi == -1 or hi <= lo:
        return None
    try:
        return json.loads(text[lo:hi + 1])
    except json.JSONDecodeError:
        return None


def call_qwen_multimodal(image_base64, prompt, model="qwen-vl-plus", max_retries=2):
    """调用 Qwen 多模态 API（视觉-语言），返回模型响应文本。
    
    参数:
        image_base64: base64 编码的图片字符串
        prompt: 文本提示
        model: 模型名称，默认 qwen-vl-plus
        max_retries: 最大重试次数
    
    返回:
        模型响应文本；失败返回空字符串
    """
    from dashscope import MultiModalConversation
    
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/png;base64,{image_base64}"},
                {"text": prompt}
            ]
        }
    ]
    
    for attempt in range(max_retries + 1):
        try:
            resp = MultiModalConversation.call(
                model=model,
                messages=messages,
                api_key=API_KEY,
            )
            if getattr(resp, "status_code", 200) == 200:
                content = resp.output.choices[0].message.content
                # qwen-vl 返回可能是 list[{'text':...}] 或 str
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict) and c.get("text"):
                            parts.append(c["text"])
                        elif isinstance(c, str):
                            parts.append(c)
                    return "\n".join(parts).strip()
                return str(content).strip()
            print(f"  [重试 {attempt + 1}] HTTP {getattr(resp, 'status_code', '?')}", file=sys.stderr)
        except Exception as e:
            print(f"  [重试 {attempt + 1}] {e}", file=sys.stderr)
        time.sleep(1.5)
    return ""
