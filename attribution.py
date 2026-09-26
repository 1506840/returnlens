# -*- coding: utf-8 -*-
"""
归因引擎 v4.2：退货工单 -> 责任方判定 + 证据 + 风险标记
新增：有凭证图片时接入 vision.describe_image，作为额外证据源；无图/失败自动降级。
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from dashscope import Generation

from vision import describe_image

load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")
CACHE_PATH = os.path.join("data", "results.jsonl")


SYSTEM_PROMPT = """你是电商售后判责专家，只做"归因"，不负责安抚用户。
仅输出一个 JSON 对象，不得有 ```json 包裹、不得有任何额外文字。

【封闭枚举】
level1 只能选：商品问题/物流问题/用户原因/商家服务/规则性
level2 只能选：
  商品问题→批次不良/功能故障/与描述不符/配件错漏发/其他
  物流问题→破损/丢件/错发/超时未达/其他
  用户原因→不会用或配置不当/买错型号规格/主观不喜欢/已激活后悔/其他
  商家服务→承诺未兑现/发货延迟/话术误导/其他
  规则性→超无理由期/已激活不支持退货/非本店商品/其他
responsible_party 只能选：供应商/仓配/运营/物流商/商家自身/用户/平台规则/无法判定

【字段】
primary_cause: {level1, level2, evidence: [原文摘录], confidence: 0~1}
secondary_cause: 同结构；无则 null
responsible_party: 字符串
actionable: 一句可执行动作，只能基于工单已有信息
needs_more_evidence: bool
ask_user: 需用户补充的信息，无则 null
risk_flag: 策略性风险描述，无则 null

【判定顺序】
第一步：判断已有证据（文字工单 + 可能的凭证图片理解）能否独立支撑责任方判定。
第二步：证据不足 → needs_more_evidence=true、填 ask_user、confidence ≤ 0.5。
第三步：证据充分 → needs_more_evidence=false、ask_user=null。
第四步：填 risk_flag（不影响责任方判定）。

【硬约束】
1. 原因与责任方是独立字段，必须分别给出。
2. evidence 必须是原文摘录（工单原文或"凭证图片理解"原文），不得推断事实。
3. 若提供了"凭证图片理解"，可将其作为证据补充；有清晰图片佐证时可适当提高 confidence，无图片佐证时不得凭空提高。
4. needs_more_evidence=true 时 confidence 必须 ≤ 0.5。
5. actionable 不得编造 SN/订单号等未出现字段。
6. 信息不足以判定时 responsible_party 填"无法判定"。

仅输出 JSON 对象本身。"""


def call_qwen(messages, model="qwen-plus", max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            resp = Generation.call(
                model=model,
                messages=messages,
                api_key=API_KEY,
                result_format="message",
                temperature=0.1,
            )
            if resp.status_code == 200:
                return resp.output.choices[0].message.content
            print(f"  [重试 {attempt + 1}] HTTP {resp.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"  [重试 {attempt + 1}] {e}", file=sys.stderr)
        time.sleep(1.5)
    return None


def extract_json(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    lo, hi = text.find("{"), text.rfind("}")
    if lo == -1 or hi == -1 or hi <= lo:
        return None
    try:
        return json.loads(text[lo:hi + 1])
    except json.JSONDecodeError:
        return None


def validate(result):
    if not isinstance(result, dict):
        return False, "结果不是对象"
    required = ["primary_cause", "responsible_party", "actionable", "needs_more_evidence"]
    missing = [k for k in required if k not in result]
    if missing:
        return False, f"缺字段: {missing}"
    valid_parties = ["供应商", "仓配", "运营", "物流商", "商家自身", "用户", "平台规则", "无法判定"]
    if result.get("responsible_party") not in valid_parties:
        return False, f"责任方非法: {result.get('responsible_party')}"
    pc = result.get("primary_cause")
    if not isinstance(pc, dict) or pc.get("level1") is None:
        return False, "primary_cause.level1 缺失"
    conf = pc.get("confidence")
    if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
        return False, f"confidence 非法: {conf}"
    if result.get("needs_more_evidence") and conf > 0.5:
        return False, f"缺证但 confidence={conf} > 0.5"
    sc = result.get("secondary_cause")
    if sc is not None and not isinstance(sc, dict):
        return False, "secondary_cause 必须是对象或 null"
    return True, "OK"


def repair(result):
    fixed = []
    if not isinstance(result, dict):
        return result
    if result.get("needs_more_evidence") and isinstance(result.get("primary_cause"), dict):
        if result["primary_cause"].get("confidence", 0) > 0.5:
            result["primary_cause"]["confidence"] = 0.5
            fixed.append("confidence 降为 0.5")
    sc = result.get("secondary_cause")
    if isinstance(sc, dict):
        sc.setdefault("level2", None)
        sc.setdefault("evidence", [])
        sc.setdefault("confidence", None)
    valid_parties = ["供应商", "仓配", "运营", "物流商", "商家自身", "用户", "平台规则", "无法判定"]
    if result.get("responsible_party") not in valid_parties:
        result["responsible_party"] = "无法判定"
        fixed.append("责任方降级")
    if not result.get("actionable"):
        result["actionable"] = "转人工复核"
        fixed.append("补充默认 actionable")
    if fixed:
        result["_repaired"] = fixed
    return result


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    cache = {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    cache[obj.get("id", "_unknown")] = obj
    except Exception:
        return {}
    return cache


def save_result(ticket_id, result):
    """写缓存时强制注入 id，保证下游能关联"""
    cache = load_cache()
    saved = {"id": ticket_id}
    if isinstance(result, dict):
        saved.update({k: v for k, v in result.items() if k != "id"})
    else:
        saved.update({"_raw": str(result), "_error": "结果不是对象"})
    cache[ticket_id] = saved
    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for v in cache.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")


def attribute(ticket, model="qwen-plus", use_cache=True):
    if use_cache:
        cached = load_cache().get(ticket["id"])
        if cached:
            print(f"  [缓存命中] {ticket['id']}", file=sys.stderr)
            return cached

    # —— 图片理解：仅当工单声明有凭证图且能解析到路径时才调用，否则优雅跳过 ——
    img_desc = None
    img_path = ticket.get("image_path")
    if ticket.get("has_image") and img_path:
        print(f"  [凭证理解] {ticket['id']} -> {img_path}", file=sys.stderr)
        img_desc = describe_image(img_path, item=ticket.get("item", ""))

    content = (
        f"工单ID:{ticket['id']} 商品:{ticket['item']} 价格:{ticket['price']} "
        f"已激活:{ticket.get('activated', False)} 物流:{ticket.get('logistics', '-')}\n"
        f"用户备注:{ticket['note']}\n聊天记录:\n"
    )
    for i, msg in enumerate(ticket.get("chat", []), 1):
        content += f"  {i}. {msg}\n"
    if img_desc:
        content += f"凭证图片理解:{img_desc}\n"
    elif ticket.get("has_image"):
        content += "凭证图片理解:（买家声称有凭证但图片缺失或不可识别）\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    raw = call_qwen(messages, model=model)

    if raw is None:
        err = {"_raw": None, "_error": "模型未返回内容"}
        save_result(ticket["id"], err)
        return err

    result = extract_json(raw)
    if result is None:
        result = {"_raw": raw, "_error": "JSON 解析失败"}
    else:
        result = repair(result)
        ok, msg = validate(result)
        if not ok:
            result["_warn"] = msg

    result["id"] = ticket["id"]
    if img_desc:
        result["_image_desc"] = img_desc   # 留痕：界面可展示图片理解结果
    save_result(ticket["id"], result)
    return result


if __name__ == "__main__":
    if not API_KEY:
        print("错误：未读到 DASHSCOPE_API_KEY，请检查 .env", file=sys.stderr)
        sys.exit(1)

    path = os.path.join("data", "tickets_seed.jsonl")
    if not os.path.exists(path):
        print(f"错误：找不到 {path}", file=sys.stderr)
        sys.exit(1)

    print("=== 归因引擎自测 ===\n", flush=True)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ticket = json.loads(line)
            print(f"----- {ticket['id']} | {ticket['item']} | ¥{ticket['price']} -----", flush=True)
            out = attribute(ticket, use_cache=False)
            print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
            print()
    print("结果已写入 data/results.jsonl")