# -*- coding: utf-8 -*-
"""
凭证图片理解：调用 qwen-vl-plus 客观描述退货凭证图片，产出可作证据的文本。
- 带本地缓存 data/vision_cache.json，避免重复调用、避免演示漂移
- 全程 try/except 降级：任何失败都返回 None，绝不阻断归因主链路
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from dashscope import MultiModalConversation

load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")
CACHE_PATH = os.path.join("data", "vision_cache.json")

VL_PROMPT = (
    "你是电商售后凭证审核助手。请客观描述这张退货凭证图片，只写你能看到的事实，"
    "不要推测责任方。重点观察：①商品是否破损/变形/有异物/使用痕迹；"
    "②商品外观与颜色，以及与常见商品图是否存在明显色差或型号差异；"
    "③能否佐证买家的退货主张（如故障现象、色差、错发）；"
    "④是否包含可识别的序列号、包装、配件缺失等信息。"
    "输出 3-5 句纯文本，不要 JSON、不要列表符号。若图片无法识别，只输出：图片不可识别。"
)


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _parse_content(msg_content):
    """qwen-vl 返回的 message.content 可能是 list[{'text':...}] 或 str，统一成 str"""
    if isinstance(msg_content, str):
        return msg_content.strip()
    if isinstance(msg_content, list):
        parts = []
        for c in msg_content:
            if isinstance(c, dict) and c.get("text"):
                parts.append(c["text"])
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts).strip()
    return str(msg_content).strip()


def describe_image(image_path, item="", use_cache=True, max_retries=2):
    """返回图片的客观描述文本；任何异常或无 key 返回 None。"""
    if not image_path or not os.path.exists(image_path):
        return None
    if not API_KEY:
        return None

    cache = _load_cache()
    key = os.path.abspath(image_path)
    if use_cache and key in cache:
        return cache[key]

    # 本地图片用 file:// 前缀；若已是 http(s) URL 则原样传
    img_ref = image_path if image_path.startswith("http") else "file://" + os.path.abspath(image_path)
    prompt_text = VL_PROMPT if not item else f"（商品：{item}）\n" + VL_PROMPT

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = MultiModalConversation.call(
                model="qwen-vl-plus",
                messages=[{
                    "role": "user",
                    "content": [{"image": img_ref}, {"text": prompt_text}],
                }],
                api_key=API_KEY,
            )
            if getattr(resp, "status_code", 200) == 200:
                text = _parse_content(resp.output.choices[0].message.content)
                if text and text != "图片不可识别":
                    cache[key] = text
                    _save_cache(cache)
                    return text
                return None
            last_err = f"HTTP {getattr(resp, 'status_code', '?')}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.5)

    print(f"  [图片理解降级] {last_err}", file=sys.stderr)
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python vision.py <图片路径> [商品名]")
        sys.exit(0)
    out = describe_image(sys.argv[1], item=sys.argv[2] if len(sys.argv) > 2 else "")
    print(out if out else "（未获得描述：无 key / 图片不存在 / 调用失败，均已降级）")