# -*- coding: utf-8 -*-
"""
从 4 篇种子论文的 arXiv e-print 源码包中抽取真实"数据图表"：
 - 解析 LaTeX figure 环境，把 \\includegraphics 与 \\caption 按环境正确配对
 - 只保留数据类图表（caption 含 accuracy/score/%/ablation/results 等），
   排除样例照片/对话演示图
 - 栅格图(png/jpg)直接复制；矢量图(.pdf)用 pymupdf 转 PNG
输出：data/figures/<pid>_<n>.png + data/figures/manifest.json
"""
import io
import json
import os
import re
import sys
import tarfile
import urllib.request

import pymupdf

PAPERS = {
    "llava15": "2310.03744",
    "som": "2310.11441",
    "mmmupro": "2409.02813",
    "viscot": "2403.16999",
}
OUT_DIR = os.path.join("data", "figures")
MIN_KB, MAX_KB = 8, 2000
IMG_EXT = (".png", ".jpg", ".jpeg", ".pdf")

# 数据图表的 caption 信号词（白名单）
DATA_WORDS = ("accuracy", "score", "result", "comparison", "performance",
              "ablation", "benchmark", "percent", "%", "distribution",
              "statistic", "quantitative", "vs.", "error bar", "breakdown",
              "f1", "bleu", "improve", "gain")
# 样例照片的 caption 信号词（黑名单，优先排除）
DEMO_WORDS = ("example", "illustrat", "qualitative", "conversation",
              "demo", "in the wild", "case study", "attention map",
              "visualization of", "we present", "our model respond")

os.makedirs(OUT_DIR, exist_ok=True)
manifest = []


def fetch_source(arxiv_id):
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchLens-demo"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def clean_caption(c):
    c = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", c)
    c = re.sub(r"\\[a-zA-Z]+\*?", " ", c)
    c = re.sub(r"[{}&~_^$%\\]", " ", c)
    return re.sub(r"\s+", " ", c).strip()


def figure_pairs(text):
    """按 figure 环境配对 (caption, [graphics文件名])"""
    pairs = []
    for blk in re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", text, re.S):
        cap_m = re.search(r"\\caption\s*(?:\[[^\]]*\])?\{(.{10,600}?)\}", blk, re.S)
        if not cap_m:
            continue
        imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", blk)
        cap = clean_caption(cap_m.group(1))
        if len(cap) > 20:
            pairs.append((cap, [i.split("/")[-1] for i in imgs]))
    return pairs


def pdf_bytes_to_png(data):
    doc = pymupdf.open(stream=data, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.2, 2.2))  # ~160dpi，控制体积
    png = pix.tobytes("png")
    doc.close()
    return png


def is_data_chart(cap):
    cl = cap.lower()
    if any(w in cl for w in DEMO_WORDS):
        return False
    return any(w in cl for w in DATA_WORDS)


for pid, aid in PAPERS.items():
    print(f"===== {pid} ({aid}) =====")
    try:
        blob = fetch_source(aid)
        tar = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
    except Exception as e:
        print(f"  ❌ 下载/解包失败: {e}")
        continue

    # 1) 收集全部 tex 的 figure 配对
    pairs = []
    for m in tar.getmembers():
        if m.name.endswith(".tex") and m.size < 800_000:
            try:
                t = tar.extractfile(m).read().decode("utf-8", errors="ignore")
            except Exception:
                continue
            pairs.extend(figure_pairs(t))

    # 2) 数据图表候选：文件名 base -> (caption, raw_bytes)
    members = {os.path.basename(m.name): m
               for m in tar.getmembers()
               if m.name.lower().endswith(IMG_EXT)
               and MIN_KB * 1024 <= m.size <= MAX_KB * 1024
               and "logo" not in m.name.lower()}

    chosen = []
    seen = set()
    for cap, img_files in pairs:
        if not is_data_chart(cap):
            continue
        for fn in img_files:
            if fn in members and fn not in seen:
                seen.add(fn)
                chosen.append((fn, cap))
    print(f"  tex figure 配对 {len(pairs)} 组，数据图表匹配 {len(chosen)} 张")

    saved = 0
    for fn, cap in chosen:
        m = members[fn]
        raw = tar.extractfile(m).read()
        if fn.lower().endswith(".pdf"):
            try:
                raw = pdf_bytes_to_png(raw)
            except Exception as e:
                print(f"  ⚠️ PDF 转换失败 {fn}: {e}")
                continue
            ext = ".png"
        else:
            ext = os.path.splitext(fn)[1].lower()
            if ext == ".jpeg":
                ext = ".jpg"
        out_name = f"{pid}_{saved+1}{ext}"
        with open(os.path.join(OUT_DIR, out_name), "wb") as f:
            f.write(raw)
        manifest.append({
            "paper_id": pid, "arxiv_id": aid,
            "file": f"data/figures/{out_name}", "orig": fn,
            "bytes": len(raw), "caption": cap[:500],
        })
        print(f"  ✅ {out_name} <- {fn} ({len(raw)//1024}KB) "
              f"{cap[:60]}...")
        saved += 1
        if saved >= 2:
            break

with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"\n共抽取 {len(manifest)} 张数据图表 → data/figures/manifest.json")
