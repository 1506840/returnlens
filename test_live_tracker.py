# -*- coding: utf-8 -*-
"""
测试 app.py 的 LiveTracker 状态机与 mine()/generate() 的回调接线。
用假的 streamlit 容器（记录 markdown 调用）驱动，不依赖 streamlit 运行时。
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


# ── 1) 通过 AST 取出 LiveTracker 类（不执行 app.py 的 streamlit 主体）────────
import ast

src = open("app.py", encoding="utf-8").read()
tree = ast.parse(src)
cls_src = None
func_src = {}
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "LiveTracker":
        cls_src = ast.get_source_segment(src, node)
    if isinstance(node, ast.FunctionDef) and node.name == "html_escape":
        func_src[node.name] = ast.get_source_segment(src, node)

assert cls_src, "app.py 中未找到 LiveTracker 类"

ns = {}
exec("import time\n" + func_src["html_escape"] + "\n" + cls_src, ns)
LiveTracker = ns["LiveTracker"]


class FakeContainer:
    """替代 st.empty()，记录每次渲染的 HTML"""
    def __init__(self):
        self.frames = []
        self.cleared = 0

    def markdown(self, html, unsafe_allow_html=False):
        self.frames.append(html)

    def empty(self):
        self.cleared += 1


def step_status(html, label):
    """判断某个步骤名在渲染结果中的状态：run/done/todo"""
    if re.search(r'rl-live-step done"><span class="rl-live-dot"></span>✓ ' + re.escape(label), html):
        return "done"
    if re.search(r'rl-live-step run"><span class="rl-live-dot"></span><span class="rl-live-ico">◌</span> ' + re.escape(label), html):
        return "run"
    if re.search(r'rl-live-step"><span class="rl-live-dot"></span>○ ' + re.escape(label), html):
        return "todo"
    return "?"


LABELS = {"load_paper": "载入论文", "calling_llm": "调用 Qwen 长上下文",
          "validating": "校验与修复", "done": "结果落盘",
          "load_gaps": "载入研究空白", "gating": "门控判定"}


def test_mine_flow():
    """测试：挖掘流的完整里程碑序列 → 全部变 done"""
    print("\n=== 测试 1: 挖掘流完整序列 ===")
    c = FakeContainer()
    t = LiveTracker(c, LiveTracker.FLOW_MINE)
    assert len(c.frames) == 1, "初始化应渲染一帧"
    assert step_status(c.frames[-1], LABELS["load_paper"]) == "todo", "初始全部待处理"

    seq = [
        ("load_paper", "论文全文加载完成（8,099 字符）"),
        ("calling_llm", "调用 qwen-plus：封闭枚举约束下抽取…"),
        ("llm_done", "模型返回（3,211 字符），解析结构化输出…"),
        ("validating", "schema 校验 → 枚举 → 置信度联动 → 证据逐字命中…"),
        ("done", "完成：3 个空白，证据命中 3/3"),
    ]
    for phase, msg in seq:
        t.on(phase, msg, {})
    final = c.frames[-1]
    for k in LiveTracker.FLOW_MINE:
        assert step_status(final, LABELS[k]) == "done", f"{k} 应为 done，实际 {step_status(final, LABELS[k])}"
    assert "3 个空白" in final, "日志应含完成信息"
    print("  ✅ 5 步全部转 done，日志含命中率")

    # 中间帧检查：calling_llm 阶段时它处于 run
    mid = c.frames[2]  # 第2次 on(calling_llm) 后的渲染
    assert step_status(mid, LABELS["calling_llm"]) == "run", "调用中步骤应处于 run 态"
    assert step_status(mid, LABELS["load_paper"]) == "done", "前序步骤应已完成"
    print("  ✅ 中间帧：前序 done、当前 run、后续 todo")


def test_hyp_flow_with_gating():
    """测试：假设生成流（含逐条 gating 事件）"""
    print("\n=== 测试 2: 假设生成流 ===")
    c = FakeContainer()
    t = LiveTracker(c, LiveTracker.FLOW_HYP)
    events = [
        ("load_gaps", "载入 4 个研究空白作为生成依据"),
        ("calling_llm", "调用 qwen-max：归纳+演绎生成候选假设并四维打分…"),
        ("parsing", "模型返回 4 条候选，开始门控判定…"),
        ("gating", "[1/4] H1 假设甲… ✅ 放行（0.802）"),
        ("gating", "[2/4] H2 假设乙… 🚫 拦截：预估工时 60 天超出预算 30 天"),
        ("gating", "[3/4] H3 假设丙… ✅ 放行（0.735）"),
        ("done", "完成：4 条假设（放行 3 / 拦截 1）"),
    ]
    for phase, msg in events:
        t.on(phase, msg, {})
    final = c.frames[-1]
    for k in LiveTracker.FLOW_HYP:
        assert step_status(final, LABELS[k]) == "done", f"{k} 应为 done"
    assert "拦截：预估工时 60 天" in final, "拦截事件应显示在日志中"
    assert "完成：4 条假设" in final
    print("  ✅ 门控逐条事件入日志，4 步全部转 done")


def test_cache_hit_shortcut():
    """测试：缓存命中时应直接全绿"""
    print("\n=== 测试 3: 缓存命中快捷路径 ===")
    c = FakeContainer()
    t = LiveTracker(c, LiveTracker.FLOW_MINE)
    t.on("cache_hit", "本地缓存命中，直接返回（llava15）", {})
    final = c.frames[-1]
    for k in LiveTracker.FLOW_MINE:
        assert step_status(final, LABELS[k]) == "done", f"缓存命中时 {k} 也应 done"
    assert "⚡" in final
    print("  ✅ 缓存命中 → 全步骤 done + ⚡ 标记")


def test_fail_state():
    """测试：失败态渲染"""
    print("\n=== 测试 4: 失败态 ===")
    c = FakeContainer()
    t = LiveTracker(c, LiveTracker.FLOW_HYP)
    t.on("load_gaps", "载入 3 个研究空白", {})
    t.on("calling_llm", "调用 qwen-max…", {})
    t.fail("LLM 调用失败")
    final = c.frames[-1]
    assert "✗ 失败：LLM 调用失败" in final
    assert "rl-live-step fail" in final
    print("  ✅ 失败卡片渲染正确")


def test_finish_summary():
    """测试：finish() 汇总行 + 耗时"""
    print("\n=== 测试 5: finish 汇总 ===")
    c = FakeContainer()
    t = LiveTracker(c, LiveTracker.FLOW_MINE)
    t.on("load_paper", "x", {})
    t.finish("总耗时 8.3 秒")
    final = c.frames[-1]
    assert "🏁 总耗时 8.3 秒" in final
    print("  ✅ finish 显示总耗时")


def test_callback_wiring():
    """测试：mine()/generate() 真实上报回调（假 API + 假缓存路径，不联网）"""
    print("\n=== 测试 6: 引擎回调接线（假 API）===")
    import tempfile, json, shutil
    import literature_mining as lm

    tmp = tempfile.mkdtemp(prefix="rl_cb_")
    try:
        # 假论文全文
        paper = {"paper_id": "test1", "title": "T", "authors_short": "A", "year": 2024,
                 "core_contribution": "C", "local_txt": os.path.join(tmp, "p.txt")}
        with open(paper["local_txt"], "w", encoding="utf-8") as f:
            f.write("Limitations include prolonged training for high-resolution images. " * 30)

        fake_json = json.dumps({
            "five_tuple": {"research_question": "q", "methodology": "m",
                           "datasets": "d", "conclusions": "c",
                           "limitations": "prolonged training for high-resolution images"},
            "gaps": [
                {"gap_id": "G1", "gap_type": "方法局限", "gap_subtype": "计算开销大",
                 "evidence": [{"quote": "prolonged training for high-resolution images",
                               "section": "Limitations"}], "confidence": 0.8},
                {"gap_id": "G2", "gap_type": "结论争议", "gap_subtype": "基线过时",
                 "evidence": [{"quote": "prolonged training for high-resolution images",
                               "section": "Limitations"}], "confidence": 0.7},
            ],
            "needs_more_evidence": False, "ask_user": None, "risk_flag": None,
        }, ensure_ascii=False)

        orig_call, orig_save, orig_cache = lm.call_qwen, lm.save_result, lm.CACHE_PATH
        lm.call_qwen = lambda messages, model="qwen-plus": fake_json
        lm.save_result = lambda pid, r: None
        lm.CACHE_PATH = os.path.join(tmp, "cache.json")

        phases = []
        r = lm.mine(paper, use_cache=False,
                    on_progress=lambda p, m, d=None: phases.append(p))
        lm.call_qwen, lm.save_result, lm.CACHE_PATH = orig_call, orig_save, orig_cache

        for expect in ["load_paper", "calling_llm", "llm_done", "validating", "done"]:
            assert expect in phases, f"mine() 应上报 {expect}，实际 {phases}"
        assert "_error" not in r
        print(f"  ✅ mine() 上报序列: {phases}")

        # generate() 回调
        import hypothesis_generator as hg
        orig_call2, orig_save2 = hg.call_qwen, hg.save_cache
        fake_hyp = json.dumps({"hypotheses": [
            {"hyp_id": "H1", "gap_id": "G1",
             "statement": "降低训练轮数可保持精度同时缩短高分辨率训练时间",
             "hypothesis_kind": "改进型",
             "rationale": "论文自述高分辨率训练迭代显著变长，若精度对训练步数不敏感则成本可大幅压缩",
             "evidence": ["prolonged training"],
             "methodology": "步数消融：40% 步数 vs 100% 步数，对比 MMMU-Pro 得分",
             "resource_tag": "单机GPU", "verifiable_via": "消融实验", "effort_days": 10,
             "scores": {k: {"value": 0.7, "evidence": "ok"} for k in
                        ["novelty", "feasibility", "impact", "evidence_strength"]}},
        ]}, ensure_ascii=False)
        hg.call_qwen = lambda messages, model="qwen-max": fake_hyp
        hg.save_cache = lambda key, hyps: None
        phases2 = []
        r2 = hg.generate(r, budget_days=30, use_cache=False,
                         on_progress=lambda p, m, d=None: phases2.append(p))
        hg.call_qwen, hg.save_cache = orig_call2, orig_save2
        for expect in ["load_gaps", "calling_llm", "parsing", "gating", "done"]:
            assert expect in phases2, f"generate() 应上报 {expect}，实际 {phases2}"
        assert len(r2) == 1 and r2[0]["gate_passed"]
        print(f"  ✅ generate() 上报序列: {phases2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("LiveTracker 分步动画 · Mock 测试")
    print("=" * 60)
    try:
        test_mine_flow()
        test_hyp_flow_with_gating()
        test_cache_hit_shortcut()
        test_fail_state()
        test_finish_summary()
        test_callback_wiring()
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
