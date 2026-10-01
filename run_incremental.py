# -*- coding: utf-8 -*-
"""
ResearchLens · 增量挖掘脚本
只挖掘新增/指定的论文，冻结已有结果保证零漂移。
架构源自 run_t004.py：冻结版兜底 → 只算目标 → 合并回写。
"""

import json
import os
import sys
import argparse

SEED = os.path.join("data", "lit_seed.jsonl")
MINING_RES = os.path.join("data", "mining_results.jsonl")
FROZEN = os.path.join("data", "mining_frozen.jsonl")


def read_jsonl(p, key_field="paper_id"):
    """读取 JSONL 文件，返回 {key: obj} 字典"""
    out = {}
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                out[o.get(key_field)] = o
    return out


def write_jsonl(p, data, seed_order):
    """按种子顺序回写 JSONL"""
    with open(p, "w", encoding="utf-8") as f:
        for pid in seed_order:
            if pid in data:
                f.write(json.dumps(data[pid], ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="ResearchLens 增量挖掘")
    parser.add_argument(
        "target",
        nargs="?",
        help="要挖掘的论文 paper_id（如 llava15）。不指定则挖掘所有未冻结的论文。"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新挖掘（忽略缓存）"
    )
    parser.add_argument(
        "--no-frozen",
        action="store_true",
        help="不使用冻结版兜底（首次运行或调试时用）"
    )
    args = parser.parse_args()

    # 1. 加载种子论文
    papers = read_jsonl(SEED)
    if not papers:
        print(f"❌ 找不到 {SEED}")
        sys.exit(1)

    seed_order = list(papers.keys())
    print(f"✅ 读到 {len(papers)} 篇种子论文: {seed_order}")

    # 2. 确定要挖掘的目标
    if args.target:
        if args.target not in papers:
            print(f"❌ 种子里没有 {args.target}")
            sys.exit(1)
        targets = [args.target]
    else:
        # 挖掘所有不在冻结版中的论文
        frozen = {} if args.no_frozen else read_jsonl(FROZEN)
        targets = [pid for pid in seed_order if pid not in frozen]
        if not targets:
            print("ℹ️ 所有论文已在冻结版中，无需增量挖掘")
            sys.exit(0)

    print(f"📌 待挖掘: {targets}")

    # 3. 执行挖掘
    from literature_mining import mine

    for pid in targets:
        paper = papers[pid]
        print(f"\n----- 挖掘 {pid} | {paper['title'][:50]} -----")
        mine(paper, use_cache=not args.force)

    # 4. 合并：冻结版兜底 + 新结果覆盖
    frozen = {} if args.no_frozen else read_jsonl(FROZEN)
    fresh = read_jsonl(MINING_RES)

    merged = dict(frozen)
    for pid in targets:
        if pid not in fresh:
            print(f"⚠️ {pid} 未写入 {MINING_RES}，跳过")
            continue
        merged[pid] = fresh[pid]

    # 5. 按种子顺序回写
    write_jsonl(MINING_RES, merged, seed_order)

    # 5b. 同步冻结基线：将本次合并结果写入 mining_frozen.jsonl，
    #     作为后续增量运行的不可变兜底，真正实现"零漂移 / 可复现"。
    #     此后未指定 --force 时，已冻结的论文直接复用，结果恒定一致。
    write_jsonl(FROZEN, merged, seed_order)
    print(f"   冻结基线已更新: {FROZEN}（{len(merged)} 篇）")

    print(f"\n✅ 完成：沿用冻结 {len(frozen)} 篇，新增/更新 {len(targets)} 篇")
    print(f"   结果已写入 {MINING_RES}")

    # 6. 打印摘要
    for pid in targets:
        if pid in merged:
            r = merged[pid]
            ft = r.get("five_tuple", {})
            gaps = r.get("gaps", [])
            print(f"\n   [{pid}] {ft.get('research_question', '?')[:60]}")
            print(f"   发现空白: {len(gaps)} 个")
            for g in gaps[:3]:
                print(f"     [{g.get('gap_id', '?')}] {g.get('gap_type', '?')}/{g.get('gap_subtype', '?')}  "
                      f"置信度={g.get('confidence', '?')}")


if __name__ == "__main__":
    main()
