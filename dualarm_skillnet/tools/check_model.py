"""Run two local Ollama environment checks using only the standard library."""

import argparse
import json
import math
from pathlib import Path
import sys
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from skillnet.decompose.prompt import build_prompt, parse_model_output
from skillnet.domains.blocks import load_problem


HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
JSON_FORMAT = {
    "type": "object",
    "properties": {
        "goal": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["goal", "steps"],
    "additionalProperties": False,
}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("本机 Ollama 接口发生重定向，已停止以保持访问范围")


def _generate(model, prompt, label, records, timeout, *, json_format=None,
              num_predict=2048):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 1101,
            "num_ctx": 8192,
            "num_predict": num_predict,
        },
    }
    if json_format is not None:
        payload["format"] = json_format
    record = {
        "test": label,
        "model": model,
        "host": HOST,
        "prompt": prompt,
        "prompt_length": len(prompt),
        "request": payload,
        "raw_output": "",
        "raw_api_response": "",
        "api_timing": {},
        "elapsed": 0.0,
        "tokens_per_second": None,
        "passed": False,
    }
    records.append(record)
    began = monotonic()
    try:
        request = Request(
            HOST + "/api/generate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            record["raw_api_response"] = response.read().decode("utf-8")
        value = json.loads(record["raw_api_response"])
        if not isinstance(value, dict):
            raise ValueError("Ollama 接口返回的 JSON 必须是对象")
        if value.get("error"):
            raise ValueError("Ollama 返回错误：" + str(value["error"]))
        text = value.get("response")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Ollama 没有返回非空模型输出")
        record["raw_output"] = text
        record["response_model"] = value.get("model", model)
        record["done"] = value.get("done")
        record["done_reason"] = value.get("done_reason")
        fields = ("total_duration", "load_duration", "prompt_eval_count",
                  "prompt_eval_duration", "eval_count", "eval_duration")
        record["api_timing"] = {field: value.get(field) for field in fields}
        duration = value.get("eval_duration")
        count = value.get("eval_count")
        if isinstance(duration, (int, float)) and duration > 0 and isinstance(count, (int, float)):
            record["tokens_per_second"] = count / (duration / 1_000_000_000)
        return text, record
    except HTTPError as error:
        record["raw_api_response"] = error.read().decode("utf-8", errors="replace")
        message = f"Ollama 接口返回 HTTP {error.code}：{record['raw_api_response']}"
        record["error"] = message
        raise ValueError(message) from error
    except (URLError, TimeoutError, OSError) as error:
        message = "未检测到本机 Ollama 服务或请求超时，请先启动 Ollama。详情：" + str(error)
        record["error"] = message
        raise ValueError(message) from error
    except (ValueError, UnicodeError) as error:
        record["error"] = str(error)
        raise
    finally:
        record["elapsed"] = monotonic() - began


def check_model(model=DEFAULT_MODEL, *, timeout=120.0):
    records = []
    result = {"model": model, "host": HOST, "passed": False, "calls": records}
    try:
        text, first = _generate(
            model, "请只用一句中文回答：一加一等于几？", "简单中文回答",
            records, timeout, num_predict=64,
        )
        if not any("\u4e00" <= character <= "\u9fff" for character in text):
            raise ValueError("简单问答未返回非空中文回答")
        first["passed"] = True
        problem = load_problem(
            PROJECT_ROOT / "tasks/s1_reach_handover.toml",
            skills=PROJECT_ROOT / "skills/blocks_v1.toml",
        )
        prompt = build_prompt(problem, mode="with_capabilities")
        text, second = _generate(
            model, prompt, "项目提示词 JSON 输出", records, timeout,
            json_format=JSON_FORMAT,
        )
        proposal = parse_model_output(text)
        if not proposal.steps:
            raise ValueError("项目 JSON 输出虽然可解析，但 steps 为空")
        second["parsed_proposal"] = {"goal": proposal.goal, "steps": proposal.steps}
        second["passed"] = True
        result["passed"] = True
    except (ValueError, OSError) as error:
        result["error"] = str(error)
        if records and not records[-1]["passed"]:
            records[-1].setdefault("error", str(error))
    result["total_elapsed"] = sum(record["elapsed"] for record in records)
    return result


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="检查本机 Ollama 的中文与项目 JSON 输出")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="已拉取的本机模型名称")
    parser.add_argument("--out", default="out/model_smoke.json", help="项目内 JSON 记录路径")
    parser.add_argument("--timeout", type=float, default=120.0, help="每次请求的超时秒数")
    args = parser.parse_args(argv)
    try:
        if not args.model.strip():
            raise ValueError("模型名称不能为空")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("请求超时必须是大于零的有限秒数")
        destination = (PROJECT_ROOT / args.out).resolve()
        if not destination.is_relative_to(PROJECT_ROOT):
            raise ValueError("冒烟记录只能写入项目目录内")
        result = check_model(args.model, timeout=args.timeout)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for record in result["calls"]:
            outcome = "通过" if record["passed"] else "失败"
            print(f"{record['test']}：{outcome}，耗时 {record['elapsed']:.3f} 秒")
            speed = record["tokens_per_second"]
            if speed is not None:
                print(f"  生成速度：{speed:.2f} token/秒")
        print(f"冒烟记录已写入 {destination}")
        if not result["passed"]:
            print("环境冒烟测试失败：" + result.get("error", "尚未通过两项检查"), file=sys.stderr)
            return 1
        print("两项环境冒烟测试均通过。")
        return 0
    except (ValueError, OSError) as error:
        print("环境冒烟测试失败：" + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
