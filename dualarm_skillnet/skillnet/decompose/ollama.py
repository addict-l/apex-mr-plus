"""Deterministic local Ollama proposals, with bounded JSON repair attempts."""

import json
import math
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .prompt import build_prompt, parse_model_output


DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_HOST = "http://127.0.0.1:11434"
JSON_FORMAT = {
    "type": "object",
    "properties": {
        "goal": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["goal", "steps"],
    "additionalProperties": False,
}


class OllamaError(ValueError):
    """An actionable model failure which the pipeline can safely report."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise OllamaError("本机 Ollama 接口返回重定向，已停止以保持本机访问范围")


class OllamaDecomposer:
    def __init__(self, model=DEFAULT_MODEL, host=DEFAULT_HOST, timeout=120,
                 temperature=0, seed=1101, mode="with_capabilities", max_retries=2):
        if not isinstance(model, str) or not model.strip():
            raise OllamaError("Ollama 模型名称不能为空")
        try:
            endpoint = urlsplit(host)
            valid_host = (
                isinstance(host, str) and not any(character.isspace() for character in host)
                and endpoint.scheme == "http" and endpoint.hostname in {"127.0.0.1", "localhost"}
                and endpoint.port is not None and 1 <= endpoint.port <= 65535
                and endpoint.username is None and endpoint.password is None
                and not endpoint.path and not endpoint.query and not endpoint.fragment
            )
        except (TypeError, ValueError, AttributeError) as error:
            raise OllamaError("Ollama 地址必须是 http://127.0.0.1:<端口> 或 http://localhost:<端口>，端口范围为 1–65535") from error
        if not valid_host:
            raise OllamaError("Ollama 地址只允许本机 http://127.0.0.1:<端口> 或 http://localhost:<端口>，不能带路径或身份信息")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise OllamaError("Ollama 请求超时必须是大于零的有限秒数")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature < 0):
            raise OllamaError("Ollama temperature 必须是非负有限数值")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise OllamaError("Ollama seed 必须是固定的非负整数")
        if mode not in {"minimal", "with_capabilities", "nl_only"}:
            raise OllamaError("Ollama 提示词模式必须是 minimal、with_capabilities 或 nl_only")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 2:
            raise OllamaError("Ollama 解析重试次数必须是 0、1 或 2")
        self.model = model
        self.host = f"http://{endpoint.hostname}:{endpoint.port}"
        self.timeout = float(timeout)
        self.temperature = temperature
        self.seed = seed
        self.mode = mode
        self.max_retries = max_retries
        self.calls: list[dict] = []
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _generate(self, prompt, attempt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": JSON_FORMAT,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_ctx": 8192, "num_predict": 2048},
        }
        call = {
            "call_index": len(self.calls) + 1,
            "model": self.model,
            "host": self.host,
            "mode": self.mode,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "raw_output": "",
            "raw_api_response": "",
            "elapsed": 0.0,
            "json_valid": False,
            "proposal_valid": False,
            "attempt": attempt,
            "retry": attempt - 1,
            "error": None,
            "api_timing": {},
            "tokens_per_second": None,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        self.calls.append(call)
        began = monotonic()
        try:
            request = Request(
                self.host + "/api/generate",
                data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            with self._opener.open(request, timeout=self.timeout) as response:
                call["http_status"] = response.status
                call["raw_api_response"] = response.read().decode("utf-8")
            try:
                body = json.loads(call["raw_api_response"])
            except ValueError as error:
                raise OllamaError("Ollama 接口响应不是合法 JSON：" + str(error)) from error
            if not isinstance(body, dict):
                raise OllamaError("Ollama 接口响应必须是 JSON 对象")
            if body.get("error"):
                raise OllamaError("Ollama 返回错误：" + str(body["error"]))
            output = body.get("response")
            if not isinstance(output, str):
                raise OllamaError("Ollama 接口响应缺少文本 response 字段")
            call["raw_output"] = output
            call["response_model"] = body.get("model", self.model)
            call["done"] = body.get("done")
            call["done_reason"] = body.get("done_reason")
            fields = ("total_duration", "load_duration", "prompt_eval_count",
                      "prompt_eval_duration", "eval_count", "eval_duration")
            call["api_timing"] = {field: body.get(field) for field in fields}
            duration = body.get("eval_duration")
            count = body.get("eval_count")
            if isinstance(duration, (int, float)) and duration > 0 and isinstance(count, (int, float)):
                call["tokens_per_second"] = count / (duration / 1_000_000_000)
            return output, call
        except HTTPError as error:
            call["http_status"] = error.code
            call["raw_api_response"] = error.read().decode("utf-8", errors="replace")
            message = f"Ollama 接口返回 HTTP {error.code}：{call['raw_api_response']}"
            call["error"] = message
            raise OllamaError(message) from error
        except OllamaError as error:
            call["error"] = str(error)
            raise
        except (URLError, TimeoutError, OSError) as error:
            message = "未检测到本机 Ollama 服务，请先启动 Ollama（" + str(error) + "）"
            call["error"] = message
            raise OllamaError(message) from error
        except UnicodeError as error:
            message = "Ollama 接口响应无法按 UTF-8 解码：" + str(error)
            call["error"] = message
            raise OllamaError(message) from error
        finally:
            call["elapsed"] = monotonic() - began

    def decompose(self, problem, feedback=None):
        base_prompt = build_prompt(problem, mode=self.mode)
        if self.mode == "nl_only":
            # Rule feedback can contain the hidden task oracle. Only the
            # success/failure signal is available to natural-language runs.
            messages = (["上一方案未完成任务，请重新从自然语言和场景自行推断 goal 与 steps。"]
                        if feedback else [])
        else:
            messages = list(feedback or ())
        for attempt in range(1, self.max_retries + 2):
            prompt = base_prompt
            if messages:
                prompt += "\n规则层与输出格式反馈：\n" + "\n".join(messages) + "\n"
            raw, call = self._generate(prompt, attempt)
            try:
                decoded = json.loads(raw)
                call["json_valid"] = True
                if not isinstance(decoded, dict):
                    raise ValueError("模型输出必须是含 goal 和 steps 的 JSON 对象")
                proposal = parse_model_output(raw)
            except ValueError as error:
                message = "模型输出解析失败：" + str(error)
                call["error"] = message
                if attempt > self.max_retries:
                    raise OllamaError(f"模型输出在重试 {self.max_retries} 次后仍无法解析：{error}") from error
                if self.mode == "nl_only":
                    # Parse details may quote model facts. Retain those only
                    # in the audit record rather than in a subsequent prompt.
                    category = "JSON 格式" if isinstance(error, json.JSONDecodeError) else "提案结构或条件语法"
                    messages.append(f"上一输出的{category}校验失败，请重新只输出含 goal 和 steps 字符串数组的 JSON 对象。")
                else:
                    messages.append(message + "。请重新只输出含 goal 和 steps 字符串数组的 JSON 对象。")
                continue
            call["proposal_valid"] = True
            proposal.instruction = getattr(problem, "instruction", "") or problem.description
            proposal.source = f"Ollama（{self.model}）"
            return proposal
        raise OllamaError("Ollama 输出重试结束但未得到方案")
