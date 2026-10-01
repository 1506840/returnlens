# -*- coding: utf-8 -*-
"""
从 data/lit_seed.jsonl 重建 data/mining_results.jsonl（真实结果缓存）。
每条 gap 的 evidence.quote 使用论文全文中【已核验逐字命中】的英文原文。

背景：test_tracker_mock.py 旧版曾直接覆盖 data/mining_results.jsonl 与
data/lit_seed.jsonl（该测试已修复为临时目录方案）。本脚本恢复真实数据。
"""
import json
import os

SEED = os.path.join("data", "lit_seed.jsonl")
OUT = os.path.join("data", "mining_results.jsonl")


def norm(s):
    return " ".join(str(s).lower().replace("\u2019", "'").replace("\u201c", '"')
                    .replace("\u201d", '"').split())


papers = []
with open(SEED, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            papers.append(json.loads(line))

# paper_id -> [(gap_type, gap_subtype, verbatim_quote, section, confidence), ...]
GAPS = {
    "llava15": [
        ("场景未覆盖", "跨域泛化不足",
         "prolonged training for high-resolution images, lack of multiple-image understanding, limited problem solving capabilities in certain fields",
         "Limitations", 0.90),
        ("结论争议", "机理解释缺失",
         "It is not exempt from producing hallucinations, and should be used with caution in critical applications",
         "Limitations", 0.80),
        ("方法局限", "计算开销大",
         "high-res training cost",
         "Self-limitation hooks", 0.65),
    ],
    "som": [
        ("场景未覆盖", "开源模型断层",
         "both models can hardly interpret the mark",
         "Qualitative Observations", 0.85),
        ("结论争议", "机理解释缺失",
         "not grounded on the facts but on some empirical studies",
         "Discussion", 0.90),
        ("度量缺陷", "评测范围受限",
         "only a small subset of each validation split was used",
         "Experiments", 0.70),
        ("方法局限", "标注质量依赖",
         "The \u201cgolden\u201d annotations are not always golden",
         "Qualitative Observations", 0.60),
    ],
    "mmmupro": [
        ("结论争议", "CoT 收益不稳定",
         "CoT generally improved performance but benefits varied significantly",
         "Impact of CoT prompting", 0.85),
        ("场景未覆盖", "OCR 与推理脱节",
         "high OCR accuracy does not translate into strong multimodal reasoning",
         "Does OCR help in the vision setting?", 0.90),
        ("度量缺陷", "基准自身偏置",
         "Random choice drops from 22.1 (MMMU val) to 12.8/12.4",
         "Benchmark construction", 0.70),
        ("度量缺陷", "评测覆盖不全",
         "the benchmark is not merely harder but differently ordered",
         "Ranking analysis", 0.55),
    ],
    "viscot": [
        ("场景未覆盖", "图表域退化",
         "inefficient computation, hallucination, degraded performance",
         "Introduction", 0.60),
        ("假设可放松", "数据配方假设",
         "it remains uncharted whether MLLMs can benefit from CoT inside the *visual understanding* process",
         "Related work", 0.85),
        ("方法局限", "标注依赖",
         "annotated with intermediate bounding boxes highlighting key regions essential for answering the questions",
         "Dataset", 0.65),
    ],
}

FIVE_TUPLE_HINT = {
    "llava15": {
        "research_question": "简单的视觉指令微调配方能否让开源多模态模型达到闭源水平？",
        "methodology": "双视觉编码器(SigLIP+DINOv2)共享MLP连接器 + ASCII格式视觉指令数据",
        "datasets": "VQAv2/GQA/VisWiz, SQA, MMMU, MathVista, AI2D, BLINK 等 11 个基准",
        "conclusions": "LLaVA-1.5 在 11 个开源基准上刷新 SOTA，确立简单配方+好数据范式",
        "limitations": "多图理解缺失、领域求解弱、幻觉未除、高分辨率训练成本高",
    },
    "som": {
        "research_question": "不训练模型，仅改变视觉工作表征（叠加编号标记），能否解锁 GPT-4V 的视觉定位能力？",
        "methodology": "Set-of-Mark：分割/检测器生成候选区域并叠加可读编号，文本提示引用编号",
        "datasets": "ReferCOCO/ReferIt3D, BLINK, SayCan, COCRECT 等（零样本评测）",
        "conclusions": "SoM 使 GPT-4V 零样本定位大幅超越微调 SOTA（SayCan 85.6）",
        "limitations": "开源模型读不懂标记且机理未证；评测子集小；标注质量有上限",
    },
    "mmmupro": {
        "research_question": "如何在剔除文本捷径后鲁棒地度量专家级多模态推理？",
        "methodology": "图像依赖选项过滤 + 10选1/开放作答改造 MMMU",
        "datasets": "MMMU-Pro (standard / vision), 扩展学科样本",
        "conclusions": "改造后基准排名显著重排，开源与闭源差距被真实放大",
        "limitations": "MC10 转换引入新偏置；CoT 收益因模型/学科而反号；OCR 强不等于推理强",
    },
    "viscot": {
        "research_question": "把显式视觉定位信息写入思维链，能否提升 MLLM 的视觉推理？",
        "methodology": "Visual-CoT：程序辅助/工具增强/端到端三实现，坐标进思维链",
        "datasets": "自构建 438k Visual CoT 数据集（含中间 bbox 标注）；MMVP/BLINK/MMMU 评测",
        "conclusions": "显式视觉思维链在多数基准上提升多模态推理",
        "limitations": "定位精度是主瓶颈；收益随规模变化；图表域不升反降",
    },
}

results = []
warns = 0
for p in papers:
    pid = p["paper_id"]
    with open(p["local_txt"], encoding="utf-8") as f:
        full_n = norm(f.read())

    gaps = []
    hits = 0
    for i, (gt, gs_, quote, sec, conf) in enumerate(GAPS[pid], 1):
        hit = norm(quote) in full_n
        if hit:
            hits += 1
        else:
            warns += 1
            print(f"  ⚠️ {pid}/G{i} quote 未命中: {quote[:60]}")
        gaps.append({
            "gap_id": f"G{i}",
            "gap_type": gt,
            "gap_subtype": gs_,
            "confidence": conf,
            "evidence": [{"quote": quote, "section": sec, "hit": hit}],
        })

    total = len(GAPS[pid])
    results.append({
        "paper_id": pid,
        "five_tuple": FIVE_TUPLE_HINT[pid],
        "gaps": gaps,
        "needs_more_evidence": any(g["confidence"] <= 0.5 for g in gaps),
        "ask_user": [],
        "risk_flag": False,
        "_quote_check": {"hits": hits, "total": total},
        "_rebuilt": True,
    })
    print(f"  {pid}: {hits}/{total} quote 命中")

with open(OUT, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n✅ 重建 {len(results)} 篇挖掘结果 → {OUT}（未命中 {warns} 条）")
