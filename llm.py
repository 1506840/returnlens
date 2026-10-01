# -*- coding: utf-8 -*-
"""
ResearchLens · LLM 调用与 JSON 解析公共层
从 attribution.py:60-96 抽出，全项目共用。
"""

import os
import sys
import re
import json
import time
import concurrent.futures
from dotenv import load_dotenv
from dashscope import Generation

load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 单次 API 调用墙钟超时（秒），避免演示时 UI 无限等待（M4）
CALL_TIMEOUT = 60

# 模型标识：默认与真实调用一致，可用环境变量覆盖（H-A）。
# 统一真源：UI 顶部 badge、文献挖掘、假设生成、图表理解共用，杜绝"宣称与实际不符"。
TEXT_MODEL = os.getenv("DASHSCOPE_TEXT_MODEL", "qwen-plus")
VL_MODEL = os.getenv("DASHSCOPE_VL_MODEL", "qwen-vl-plus")

# 复用线程池：避免每次调用都新建 executor（L-B）。max_workers 取 4，
# 单次超时未返回的任务只占一个槽位，不会阻塞后续调用（本项目串行调用为主）。
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _safe_call(fn, timeout, *args, **kwargs):
    """在线程中执行 fn，超时或异常统一返回 None，绝不向上抛。"""
    fut = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print(f"  [超时] API 调用超过 {timeout}s", file=sys.stderr)
        return None
    except Exception as e:  # noqa: BLE001
        print(f"  [异常] {e}", file=sys.stderr)
        return None


def call_qwen(messages, model=TEXT_MODEL, max_retries=2, timeout=CALL_TIMEOUT):
    """调用 Qwen API，返回原始文本；失败/超时返回 None。

    任何一次调用超时被线程池截断（M4），并按指数退避重试。
    """
    for attempt in range(max_retries + 1):
        resp = _safe_call(
            Generation.call, timeout,
            model=model, messages=messages, api_key=API_KEY,
            result_format="message", temperature=0.1,
        )
        if resp is None:
            print(f"  [重试 {attempt + 1}] 调用失败/超时", file=sys.stderr)
        elif getattr(resp, "status_code", None) == 200:
            return resp.output.choices[0].message.content
        else:
            print(f"  [重试 {attempt + 1}] HTTP {getattr(resp, 'status_code', '?')}", file=sys.stderr)
        time.sleep(min(1.5 * (2 ** attempt), 8))
    return None


def extract_json(text):
    """从模型输出中提取 JSON 对象；失败返回 None。

    解析顺序：① 直接 json.loads → ② 去 ```json 围栏 → ③ 取最外层平衡花括号块
    （用括号深度匹配，避免模型在 JSON 外写带 } 的文字导致截断）。
    """
    if not text:
        return None
    text = text.strip()
    # ① 直接解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # ② 去 ```json / ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
        text = candidate
    # ③ 取最外层平衡花括号块
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def call_qwen_multimodal(image_base64, prompt, model=VL_MODEL,
                         max_retries=2, timeout=CALL_TIMEOUT):
    """调用 Qwen 多模态 API（视觉-语言），返回模型响应文本。

    参数:
        image_base64: base64 编码的图片字符串
        prompt: 文本提示
        model: 模型名称，默认 qwen-vl-plus
        max_retries: 最大重试次数
        timeout: 单次调用墙钟超时（秒）

    返回:
        模型响应文本；失败/超时返回 None（与 call_qwen 统一语义，M4）
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
        resp = _safe_call(
            MultiModalConversation.call, timeout,
            model=model, messages=messages, api_key=API_KEY,
        )
        if resp is None:
            print(f"  [重试 {attempt + 1}] 调用失败/超时", file=sys.stderr)
        elif getattr(resp, "status_code", None) == 200:
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
        else:
            print(f"  [重试 {attempt + 1}] HTTP {getattr(resp, 'status_code', '?')}", file=sys.stderr)
        time.sleep(min(1.5 * (2 ** attempt), 8))
    return None
