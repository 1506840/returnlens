# -*- coding: utf-8 -*-
"""
ResearchLens · 阶段1 API 联通实测
与 check.py 的区别：走 llm.call_qwen / extract_json 公共层，验证真实调用链路，
并顺带确认 taxonomy 枚举能被模型按封闭集输出（为阶段2 抽取做准备）。

注意：沙箱会注入指向本机 9 端口的代理占位地址导致请求被本地拒绝，
故本脚本在导入 dashscope 前主动清空 *_PROXY。
用法：python check_api.py
"""

import os

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_v, None)

import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
import taxonomy as T
from llm import call_qwen, extract_json

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def test_roundtrip():
    """1) 最小往返 + JSON 解析"""
    out = call_qwen([
        {"role": "system", "content": "你只输出 JSON，不要任何解释。"},
        {"role": "user", "content": '输出 {"ok": true, "n": 7}'},
    ])
    js = extract_json(out)
    ok = isinstance(js, dict) and js.get("ok") is True and js.get("n") == 7
    print(f"[1] call_qwen + extract_json 往返: {'OK' if ok else 'FAIL'} -> {out!r}")
    return ok


def test_enum_closure():
    """2) 让模型从 GAP_TYPE_LIST 封闭集中选一个，验证枚举可约束输出"""
    choices = T.GAP_TYPE_LIST
    out = call_qwen([
        {"role": "system", "content": "你必须且只能从给定列表中选择一个原词作答，不要解释。"},
        {"role": "user", "content": (
            f"列表：{json.dumps(choices, ensure_ascii=False)}\n"
            "论文自述「不支持多图理解」属于列表中的哪一类？只回答一个词。"
        )},
    ] )
    ans = (out or "").strip().strip('"').strip()
    ok = ans in choices
    print(f"[2] taxonomy 封闭枚举可约束输出: {'OK' if ok else 'FAIL'} -> {ans!r}")
    return ok


if __name__ == "__main__":
    r1 = test_roundtrip()
    r2 = test_enum_closure()
    if r1 and r2:
        print("\nAPI 联通与枚举约束均通过，阶段1 验收完成。")
        sys.exit(0)
    print("\n存在未通过项，见上方标记。")
    sys.exit(1)
