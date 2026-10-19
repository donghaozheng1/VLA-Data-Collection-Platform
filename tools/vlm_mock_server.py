"""OpenAI 兼容的 VLM **模拟服务**（纯标准库），用于在没有 Qwen3-VL 权重时验证整条链路。

与 `tools/infer_service.py` 同一套路：独立进程 + 纯标准库 http.server，
不引入任何依赖，只监听 127.0.0.1。平台侧把 `vlm_config.json` 的 `base_url`
指向它，就能把「抽帧 → 拼图 → 提问 → 解析 → 写 qc_findings → 打回重采」整条链路跑通，
而且**结果是确定的**，可以拿来写回归测试。

用法：

    python tools/vlm_mock_server.py --port 8199 --mode mixed

`--mode` 决定它怎么回答：

    all_pass     每条都合规（用来验证"不会误报"）
    all_fail     每条都不合规（用来验证 warning 能落库、能进打回重采）
    mixed        按问题文本哈希交替 yes/no（默认）
    disagree     正反两次都答 yes（用来验证"正反不一致 → 待人工确认"）
    contradict   判别式说"没完成"、描述式说"pass"（用来验证跨模式矛盾会被标出）
    unparseable  返回非 JSON 废话（用来验证解析兜底与「未取到回答」路径）
    slow         每次延迟 1.5s（用来验证进度上报与超时）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = "mixed"
MODEL = "qwen3-vl:4b-mock"

#: 每个问题项（按 1 开始的编号）的**合规答案**，由测试按当前问题清单设置。
#: 为什么需要它：`all_pass` 不能假设「正向恒答 yes」—— 清单里一旦出现
#: `expect: "no"` 的项（例如「有没有人经过」「开放世界事件」），
#: 「全部答 yes」反而意味着**这些项全都不合规**，all_pass 会变成 all_fail。
EXPECTS: list[str] = []


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _flip(value: str) -> str:
    return "no" if value == "yes" else "yes"


def _expect_of(qid: str) -> str:
    """这个问题的合规答案：正向问法用该项的 expect，反向问法用它的反面。"""
    negative = qid.endswith("n")
    digits = qid[:-1] if negative else qid
    index = int(digits) if digits.isdigit() else 1
    expect = EXPECTS[index - 1] if 0 <= index - 1 < len(EXPECTS) else "yes"
    return _flip(expect) if negative else expect


def _answer_for(question: str, seed_source: str = "") -> str:
    """按（画面内容 + 问题）取确定答案。

    必须把**画面**也算进去：只看问题文本的话，同一数据集里所有 episode 会拿到
    一模一样的回答（问题都一样），演示和测试都区分不出好坏样本。
    """
    return "yes" if _seed(f"{seed_source}|{question}") % 2 == 0 else "no"


def _ids_and_questions(prompt: str) -> list[tuple[str, str]]:
    """从提示词里抠出 `1. 问题…` 形式的编号问题（模拟版专用，真实模型不需要）。"""
    out: list[tuple[str, str]] = []
    for line in prompt.splitlines():
        match = re.match(r"^\s*(\d+n?)\.\s*(.+?)\s*$", line)
        if match:
            out.append((match.group(1), match.group(2)))
    return out


def _checklist_reply(prompt: str, seed_source: str = "") -> str:
    pairs = _ids_and_questions(prompt)
    if not pairs:                                  # 抠不到问题就退化成固定两条
        pairs = [("1", "任务是否完成？"), ("1n", "任务是否未完成？")]

    answers = []
    for qid, question in pairs:
        negative = qid.endswith("n")
        if MODE == "all_pass":
            # 合规 = 正向答该项的 expect、反向答它的反面（expect 可能是 no）
            answer = _expect_of(qid)
        elif MODE in ("all_fail", "contradict"):
            # contradict：判别式一律"没完成"，而描述式一律 pass —— 故意制造跨模式矛盾
            answer = _flip(_expect_of(qid))
        elif MODE == "disagree":
            answer = "yes"                            # 正反都答 yes → 必然不一致
        else:  # mixed / slow / unparseable 都按（画面 + 问题）哈希
            answer = _answer_for(question, seed_source)
        answers.append({"id": qid, "answer": answer,
                        "why": f"mock({MODE})：依据拼图末帧判断"})
    return json.dumps({"answers": answers, "overall": f"mock 总体印象（{MODE}）"},
                      ensure_ascii=False)


def _describe_reply(prompt: str, seed_source: str = "") -> str:
    if MODE == "all_fail":
        verdict, completed, signs = "fail", False, ["末帧目标物体仍在原位", "机械臂停在半空"]
    elif MODE in ("all_pass", "contradict"):
        verdict, completed, signs = "pass", True, []
    elif MODE == "disagree":
        verdict, completed, signs = "pass", False, []      # 故意自相矛盾
    else:
        failed = _seed(f"{seed_source}|describe") % 2 == 1
        verdict, completed = ("fail", False) if failed else ("pass", True)
        signs = ["末帧目标物体未到位"] if failed else []
    return json.dumps({
        "description": "初始状态：目标物体在工作台左侧，机械臂位于初始位。"
                       "动作：机械臂下降抓取后移动到目标位置上方。"
                       "结束状态：物体位于目标区域，机械臂抬起待命。",
        "suggested_instruction": "把目标物体抓取并放到指定位置",
        "task_completed": completed,
        "failure_signs": signs,
        "verdict": verdict,
        "reason": f"mock({MODE})：依据首末帧对比得出",
    }, ensure_ascii=False)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - 静音默认日志
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 约定
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list",
                        "data": [{"id": MODEL, "object": "model", "owned_by": "mock"},
                                 {"id": "qwen3-vl:4b", "object": "model", "owned_by": "mock"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:  # noqa: BLE001
            self._send({"error": "bad json"}, 400)
            return

        images = 0
        prompt = ""
        image_hash = ""
        for message in payload.get("messages") or []:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        prompt += part.get("text") or ""
                    elif part.get("type") == "image_url":
                        images += 1
                        if not image_hash:
                            url = (part.get("image_url") or {}).get("url") or ""
                            # 用画面本身的哈希当种子，让不同 episode 得到不同判定
                            image_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        seed_source = f"{image_hash}"
        if MODE == "slow":
            time.sleep(1.5)

        if MODE == "unparseable":
            text = "抱歉，我看不出这条数据有什么问题。（mock 故意返回非 JSON）"
        elif "answers" in prompt and "verdict" in prompt:
            text = _checklist_reply(prompt, seed_source) + "\n" + _describe_reply(prompt, seed_source)
        elif "answers" in prompt:
            text = _checklist_reply(prompt, seed_source)
        elif "verdict" in prompt:
            text = _describe_reply(prompt, seed_source)
        else:
            text = _describe_reply(prompt, seed_source)

        self._send({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "images": images},
        })


def main() -> None:
    global MODE, MODEL
    parser = argparse.ArgumentParser(description="OpenAI 兼容的 VLM 模拟服务（用于离线验证）")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mode", default="mixed",
                        choices=["all_pass", "all_fail", "mixed", "disagree", "contradict",
                                 "unparseable", "slow"])
    parser.add_argument("--model", default="qwen3-vl:4b-mock")
    args = parser.parse_args()
    MODE, MODEL = args.mode, args.model
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[vlm-mock] http://{args.host}:{args.port}/v1  mode={MODE}  model={MODEL}")
    print("[vlm-mock] Ctrl+C 退出")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[vlm-mock] 已停止")


if __name__ == "__main__":
    main()
