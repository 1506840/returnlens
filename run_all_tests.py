# -*- coding: utf-8 -*-
"""
ResearchLens · 全量验收脚本
一键运行所有 mock 测试（不依赖 API），确认系统核心逻辑全部可用。
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(__file__))

TESTS = [
    ("文献挖掘引擎", "test_mining_mock.py"),
    ("假设生成引擎", "test_hypothesis_mock.py"),
    ("图表理解引擎", "test_chart_mock.py"),
    ("研究进展追踪", "test_tracker_mock.py"),
    ("分步动画引擎", "test_live_tracker.py"),
    ("证据回链可追溯", "test_backlink.py"),
    ("答辩模式开关", "test_stage_mode.py"),
]

print("=" * 60)
print("ResearchLens · 全量验收")
print("=" * 60)

results = []
total_pass = 0
total_fail = 0

for name, script in TESTS:
    print(f"\n{'─' * 60}")
    print(f"▶ {name} ({script})")
    print(f"{'─' * 60}")
    ret = subprocess.run([sys.executable, script], capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    print(ret.stdout or "")
    if ret.stderr:
        # 只显示关键 stderr，过滤掉 streamlit 警告
        for line in ret.stderr.split("\n"):
            if "error" in line.lower() or "traceback" in line.lower():
                print(f"  [stderr] {line}")
    passed = ret.returncode == 0
    results.append((name, passed))
    if passed:
        total_pass += 1
    else:
        total_fail += 1

print("\n" + "=" * 60)
print("验收汇总")
print("=" * 60)
for name, passed in results:
    status = "✅ 通过" if passed else "❌ 失败"
    print(f"  {status}  {name}")

print(f"\n  通过: {total_pass}  失败: {total_fail}  总计: {total_pass + total_fail}")

if total_fail:
    print("\n  ⚠️ 有失败项，请检查后重试")
    sys.exit(1)
else:
    print("\n  🎉 ResearchLens 全量验收通过！")

# 附加检查：文件完整性
print("\n" + "=" * 60)
print("文件完整性检查")
print("=" * 60)

required_files = [
    "llm.py",
    "taxonomy.py",
    "literature_mining.py",
    "hypothesis_generator.py",
    "research_tracker.py",
    "chart_understanding.py",
    "run_incremental.py",
    "app.py",
    "README.md",
    "答辩手卡.md",
    "data/lit_seed.jsonl",
]

for f in required_files:
    exists = os.path.exists(f)
    status = "✅" if exists else "❌"
    print(f"  {status} {f}")
