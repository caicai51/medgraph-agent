"""Run a small real end-to-end SSE generation regression against the local API."""
import json
import time
import urllib.request
from pathlib import Path


CASES = [
    ("gen_01", "感冒有哪些常见的处理方法？"),
    ("gen_02", "高血压患者日常生活中应注意什么？"),
    ("gen_03", "糖尿病常见症状有哪些？"),
    ("gen_04", "感冒时通常需要做哪些检查？"),
    ("gen_05", "火星感冒药应该怎么服用？"),
]


def run_case(case_id, query):
    body = json.dumps({"query": query, "user_id": "local-e2e", "use_local": False, "model": "qwen-plus"}).encode("utf-8")
    request = urllib.request.Request("http://127.0.0.1:8000/stream", data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    events, event_type, data_lines = [], None, []
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_lines.append(line[6:])
            elif not line and data_lines:
                payload = "\n".join(data_lines)
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {"raw": payload}
                events.append({"event": event_type or "message", "data": payload})
                event_type, data_lines = None, []
    answer = "".join(item["data"].get("content", "") for item in events if item["event"] == "token")
    done = next((item["data"] for item in events if item["event"] == "done"), {})
    errors = [item["data"] for item in events if item["event"] == "error"]
    return {"case_id": case_id, "query": query, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "answer": answer, "answer_characters": len(answer), "done": done, "errors": errors, "events": events}


def main():
    output = {"cases": []}
    for case_id, query in CASES:
        try:
            output["cases"].append(run_case(case_id, query))
        except Exception as exc:
            output["cases"].append({"case_id": case_id, "query": query, "errors": [{"type": type(exc).__name__}], "answer_characters": 0})
    output["summary"] = {"total": len(output["cases"]),
                         "nonempty_answers": sum(item["answer_characters"] > 0 for item in output["cases"]),
                         "error_cases": sum(bool(item.get("errors")) for item in output["cases"])}
    Path("outputs/generation_regression.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
