# -*- coding: utf-8 -*-
"""只归因 T004，并把 T001-T003 从冻结版逐字恢复，保证零漂移。"""
import json, os

SEED = os.path.join("data", "tickets_seed.jsonl")
RES = os.path.join("data", "results.jsonl")
FROZEN = os.path.join("data", "results_frozen.jsonl")
TARGET = "T004"

def read_jsonl(p):
    out = {}
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                out[o.get("id")] = o
    return out

# 1. 冻结版兜底数据
frozen = read_jsonl(FROZEN)
if not frozen:
    raise SystemExit("❌ 缺 results_frozen.jsonl，先执行备份")

# 2. 只跑 T004
tickets = read_jsonl(SEED)
if TARGET not in tickets:
    raise SystemExit(f"❌ 种子里没有 {TARGET}，检查 tickets_seed.jsonl")

from attribution import attribute
attribute(tickets[TARGET], use_cache=False)   # 只算这一条

# 3. 以冻结版为底，只覆盖/新增 T004
merged = dict(frozen)
fresh = read_jsonl(RES)
if TARGET not in fresh:
    raise SystemExit("❌ T004 没写进 results.jsonl，检查 attribution 返回")
merged[TARGET] = fresh[TARGET]

# 4. 按种子顺序回写
with open(RES, "w", encoding="utf-8") as f:
    for tid in tickets:
        if tid in merged:
            f.write(json.dumps(merged[tid], ensure_ascii=False) + "\n")

print(f"✅ 完成：沿用冻结 {len(frozen)} 条，新增/更新 {TARGET}")
r = merged[TARGET]
pc = r.get("primary_cause") or {}
print(f"   {TARGET} 主因: {pc.get('level1')}/{pc.get('level2')}")
print(f"   {TARGET} 责任方: {r.get('responsible_party')}  置信度: {pc.get('confidence')}")
print(f"   图片理解: {r.get('_image_desc','（无）')[:120]}")