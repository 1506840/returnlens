# -*- coding: utf-8 -*-
"""一次性修复：给 results.jsonl 补 id 字段"""
import json

IDS = ["T001", "T002", "T003"]
PATH = "data/results.jsonl"

rows = []
with open(PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"读到 {len(rows)} 行, 原 id: {[r.get('id') for r in rows]}")

with open(PATH, "w", encoding="utf-8") as f:
    for i, row in enumerate(rows):
        row["id"] = IDS[i] if i < len(IDS) else f"T{len(rows)+1}"
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"已写入: {[r['id'] for r in rows]}")