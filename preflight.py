# -*- coding: utf-8 -*-
"""
ResearchLens · 答辩模式离线自检（preflight）
纯本地检查，零 API、零联网。逐项验证"断网也能完整演示"的全部前提，
供 app.py 侧边栏「答辩模式」开关驱动红绿灯面板。

检查项：
  P1 种子论文完整（≥4 篇，字段齐）
  P2 论文全文可用（本地文件存在且 ≥3000 字符）
  P3 挖掘缓存覆盖（每篇都有结果、无错误、空白 ≥2）
  P4 证据闭环（每条空白引用复核仍能在论文全文逐字命中）
  P5 假设缓存覆盖（每篇都有假设，含放行与拦截样例）
  P6 回链可追溯（每条假设的 gap 都能解析到带证据的空白卡）
"""

import json
import os
import re
from io_jsonl import read_jsonl

SEED_PATH = os.path.join("data", "lit_seed.jsonl")
MINING_PATH = os.path.join("data", "mining_results.jsonl")
HYP_PATH = os.path.join("data", "hypotheses.jsonl")

MIN_PAPERS = 4
MIN_TEXT = 3000


def _norm(s):
    """与 rebuild_seed_cache / test_backlink 一致的归一化：弯引号→直引号，空白折叠，小写"""
    s = str(s).replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(s.lower().split())


def _load_jsonl(path, key):
    # 统一走 io_jsonl（M-B）
    return read_jsonl(path, key=key)


def run_preflight():
    """执行全部自检，返回 {"all_ok": bool, "checks": [(id, name, ok, detail)]}"""
    checks = []

    def add(cid, name, ok, detail=""):
        checks.append((cid, name, bool(ok), detail))

    # ── P1 种子论文 ────────────────────────────────────────────
    seeds = _load_jsonl(SEED_PATH, "paper_id")
    seed_ok = len(seeds) >= MIN_PAPERS
    missing_fields = []
    for pid, p in seeds.items():
        for field in ("title", "year", "local_txt", "core_contribution"):
            if not p.get(field):
                missing_fields.append(f"{pid}.{field}")
    add("P1", "种子论文完整", seed_ok and not missing_fields,
        f"{len(seeds)} 篇" + (f"；缺字段 {missing_fields}" if missing_fields else ""))

    # ── P2 论文全文 ────────────────────────────────────────────
    texts = {}
    short = []
    for pid, p in seeds.items():
        path = p.get("local_txt", "")
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                texts[pid] = _norm(f.read())
            if len(texts[pid]) < MIN_TEXT:
                short.append(pid)
        else:
            short.append(f"{pid}(文件缺失)")
    add("P2", "论文全文本地可用", len(texts) == len(seeds) and not short,
        f"{len(texts)}/{len(seeds)} 篇已加载" + (f"；异常 {short}" if short else "（零联网）"))

    # ── P3 挖掘缓存覆盖 ────────────────────────────────────────
    mining = _load_jsonl(MINING_PATH, "paper_id")
    errs = [pid for pid, r in mining.items() if "_error" in r]
    thin = [pid for pid, r in mining.items() if len(r.get("gaps", [])) < 2]
    cover = set(seeds) <= set(mining)
    add("P3", "挖掘缓存覆盖", cover and not errs and not thin,
        f"{len(mining)}/{len(seeds)} 篇有结果"
        + (f"；错误 {errs}" if errs else "") + (f"；空白<2 {thin}" if thin else ""))

    # ── P4 证据闭环（独立复核，不信任 _quote_check）────────────
    quote_total = quote_hit = 0
    bad = []
    for pid, doc in mining.items():
        full = texts.get(pid, "")
        for g in doc.get("gaps", []):
            for ev in (g.get("evidence") or []):
                q = ev.get("quote") if isinstance(ev, dict) else None
                if not q:
                    continue
                quote_total += 1
                if _norm(q) in full:
                    quote_hit += 1
                else:
                    bad.append(f"{pid}/{g.get('gap_id')}")
    add("P4", "证据逐字命中原文", quote_total > 0 and quote_hit == quote_total,
        f"{quote_hit}/{quote_total} 条引用复核通过" + (f"；未命中 {bad}" if bad else ""))

    # ── P5 假设缓存覆盖 ────────────────────────────────────────
    hyp_cache = _load_jsonl(HYP_PATH, "cache_key")
    hyp_by_pid = {}
    for key, obj in hyp_cache.items():
        pid = re.match(r"(.+)_budget", key)
        pid = pid.group(1) if pid else key
        lst = obj.get("hypotheses", [])
        if lst:
            prev = hyp_by_pid.get(pid, [])
            hyp_by_pid[pid] = prev + lst
    no_hyp = [pid for pid in seeds if not hyp_by_pid.get(pid)]
    any_pass = any(h.get("gate_passed") for lst in hyp_by_pid.values() for h in lst)
    add("P5", "假设缓存覆盖", not no_hyp and any_pass,
        f"{sum(len(v) for v in hyp_by_pid.values())} 条假设 / {len(hyp_by_pid)} 篇"
        + (f"；缺假设 {no_hyp}" if no_hyp else "")
        + ("" if any_pass else "；⚠️ 无放行样例"))

    # ── P6 回链可追溯 ──────────────────────────────────────────
    unlinked = []
    n_hyp = 0
    for pid, lst in hyp_by_pid.items():
        gidx = {g.get("gap_id") for g in (mining.get(pid, {}).get("gaps", []))}
        for h in lst:
            n_hyp += 1
            gid = h.get("gap_id") or ""
            tg = h.get("target_gap_ids")
            if isinstance(tg, list) and tg:
                gid = str(tg[0])
            elif tg and not gid:
                gid = str(tg)
            if gid not in gidx:
                unlinked.append(f"{pid}/{h.get('hyp_id')}→{gid!r}")
    add("P6", "假设回链可追溯", n_hyp > 0 and not unlinked,
        f"{n_hyp - len(unlinked)}/{n_hyp} 条可解析"
        + (f"；断链 {unlinked}" if unlinked else ""))

    all_ok = all(ok for _, _, ok, _ in checks)
    return {"all_ok": all_ok, "checks": checks}


if __name__ == "__main__":
    r = run_preflight()
    print("=" * 62)
    print("ResearchLens · 答辩模式离线自检")
    print("=" * 62)
    for cid, name, ok, detail in r["checks"]:
        print(f"  {'🟢' if ok else '🔴'} {cid} {name:<12} {detail}")
    print("-" * 62)
    print(("🟢 断网演示就绪：可放心上台" if r["all_ok"] else "🔴 存在未通过项，先修复再演示"))
    raise SystemExit(0 if r["all_ok"] else 1)
