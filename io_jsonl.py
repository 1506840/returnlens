# -*- coding: utf-8 -*-
"""
ResearchLens · JSONL 读写公共层（M-B）
消除 literature_mining / hypothesis_generator / research_tracker / preflight
四处重复的逐行解析实现，统一为单一真源。

- read_jsonl(path, key=None)
    key=None  -> 返回对象 list
    key 给定  -> 返回 {obj[key]: obj} 字典（缺失 key 的对象被跳过）
- write_jsonl(path, records)
    覆盖写入：records 为可迭代的 dict 序列
- load_jsonl_cache(path, key)
    等价于 read_jsonl(path, key=key)，保留旧 load_cache 语义
"""

import json
import os


def read_jsonl(path, key=None):
    if not os.path.exists(path):
        return {} if key else []
    out = {} if key else []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if key is not None:
                k = obj.get(key)
                if k is not None:
                    out[k] = obj
            else:
                out.append(obj)
    return out


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl_cache(path, key):
    return read_jsonl(path, key=key)
