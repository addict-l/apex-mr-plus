"""Exercise the model adapter with a strictly local HTTP simulation."""

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
from threading import Thread
import unittest

from skillnet.decompose.ollama import OllamaDecomposer, OllamaError
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run


VALID_OUTPUT = json.dumps({
    "goal": ["at(red, board_l1)"],
    "steps": ["pick(B, red, stock_r1)", "handover(B, A, red, hand_mid)",
              "place(A, red, board_l1)"],
}, ensure_ascii=False)


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def unused_endpoint():
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        if port != 11434:
            return f"http://127.0.0.1:{port}"


class SimulatedOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, payload))
        response = self.server.responses.popleft()
        status, headers, body = response
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body, ensure_ascii=False).encode("utf-8"))

    def log_message(self, *args):
        pass


def response(text=VALID_OUTPUT):
    return 200, {}, {
        "model": "qwen2.5:7b-instruct", "response": text, "done": True,
        "total_duration": 300_000_000, "load_duration": 10_000_000,
        "prompt_eval_count": 100, "prompt_eval_duration": 100_000_000,
        "eval_count": 20, "eval_duration": 200_000_000,
    }


class OllamaAdapterTests(unittest.TestCase):
    def setUp(self):
        self.server = LocalServer(("127.0.0.1", 0), SimulatedOllama)
        self.assertNotEqual(self.server.server_address[1], 11434)
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.server.requests = []
        self.server.responses = deque()
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.problem = load_problem("tasks/s1_reach_handover.toml")

    def stop_server(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2)
            self.server = None

    def test_success_uses_schema_and_records_full_call_evidence(self):
        self.server.responses.append(response())
        adapter = OllamaDecomposer(host=self.host)
        proposal = adapter.decompose(self.problem)
        self.assertEqual(proposal.goal, ["at(red, board_l1)"])
        self.assertEqual(len(proposal.steps), 3)
        self.assertIn("Ollama", proposal.source)
        self.assertIn(adapter.model, proposal.source)
        path, payload = self.server.requests[0]
        self.assertEqual(path, "/api/generate")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "qwen2.5:7b-instruct")
        self.assertEqual(payload["options"], {"temperature": 0, "seed": 1101,
                                             "num_ctx": 8192, "num_predict": 2048})
        self.assertEqual(payload["format"]["required"], ["goal", "steps"])
        self.assertEqual(payload["format"]["properties"]["steps"]["items"], {"type": "string"})
        self.assertEqual(len(adapter.calls), 1)
        call = adapter.calls[0]
        self.assertTrue(call["json_valid"])
        self.assertEqual(call["raw_output"], VALID_OUTPUT)
        self.assertEqual(call["prompt"], payload["prompt"])
        self.assertEqual(call["prompt_length"], len(payload["prompt"]))
        self.assertEqual(call["model"], adapter.model)
        self.assertGreaterEqual(call["elapsed"], 0)
        self.assertEqual(json.loads(call["raw_api_response"])["response"], VALID_OUTPUT)
        self.assertEqual(call["api_timing"]["eval_count"], 20)
        self.assertAlmostEqual(call["tokens_per_second"], 100)

    def test_invalid_json_retries_with_the_error_before_succeeding(self):
        self.server.responses.extend([response("不是JSON"), response()])
        adapter = OllamaDecomposer(host=self.host)
        proposal = adapter.decompose(self.problem, ["请补齐红色积木的位置目标"])
        self.assertEqual(len(proposal.steps), 3)
        self.assertEqual(len(self.server.requests), 2)
        self.assertIn("请补齐红色积木的位置目标", self.server.requests[0][1]["prompt"])
        self.assertIn(adapter.calls[0]["error"], self.server.requests[1][1]["prompt"])
        self.assertFalse(adapter.calls[0]["json_valid"])
        self.assertTrue(adapter.calls[1]["json_valid"])
        self.assertEqual([call["attempt"] for call in adapter.calls], [1, 2])
        self.assertEqual([call["retry"] for call in adapter.calls], [0, 1])

    def test_nl_only_rule_feedback_does_not_reveal_formal_task_goals(self):
        self.server.responses.append(response())
        adapter = OllamaDecomposer(host=self.host, mode="nl_only")
        feedback = ["仍未解决的目标：free(A), free(B), at(red, board_l1)",
                    "为满足 place(A, red, board_l1) 的条件：held(A, red)"]
        adapter.decompose(self.problem, feedback)
        prompt = self.server.requests[0][1]["prompt"]
        for text in feedback:
            self.assertNotIn(text, prompt)
        for goal in self.problem.goal:
            self.assertNotIn(str(goal), prompt)
        self.assertNotIn("held(A, red)", prompt)

    def test_nl_only_parse_retry_keeps_model_goal_facts_only_in_audit(self):
        invalid = json.dumps({"goal": ["at(red, board_l1) extra"], "steps": []})
        self.server.responses.extend([response(invalid), response()])
        adapter = OllamaDecomposer(host=self.host, mode="nl_only")
        adapter.decompose(self.problem)
        self.assertEqual(len(self.server.requests), 2)
        self.assertIn("at(red, board_l1)", adapter.calls[0]["error"])
        retried_prompt = self.server.requests[1][1]["prompt"]
        for goal in self.problem.goal:
            self.assertNotIn(str(goal), retried_prompt)
        self.assertNotIn(adapter.calls[0]["error"], retried_prompt)
        self.assertNotEqual(self.server.requests[0][1]["prompt"], retried_prompt)

    def test_three_invalid_outputs_raise_a_chinese_error_and_stop(self):
        self.server.responses.extend([response("{")] * 3)
        adapter = OllamaDecomposer(host=self.host)
        with self.assertRaisesRegex(OllamaError, "重试|解析|JSON"):
            adapter.decompose(self.problem)
        self.assertEqual(len(self.server.requests), 3)
        self.assertEqual(len(adapter.calls), 3)
        self.assertTrue(all(not call["json_valid"] and call["error"] for call in adapter.calls))
        self.assertEqual(adapter.calls[-1]["retry"], 2)

    def test_exhausted_json_repairs_fall_back_to_the_same_optimal_search(self):
        self.problem.proposal = None
        baseline = run(self.problem)
        self.server.responses.extend([response("{")] * 3)
        adapter = OllamaDecomposer(host=self.host)
        result = run(self.problem, decomposer=adapter)
        self.assertEqual(result.status, "solved")
        self.assertIsNone(result.proposal)
        self.assertIsNone(result.completion)
        self.assertLessEqual(abs(result.schedule.makespan - baseline.schedule.makespan), 1e-6)
        self.assertEqual(len(self.server.requests), 3)
        self.assertEqual(len(result.model_calls), 3)
        self.assertTrue(result.model_error)
        self.assertTrue(any("退回不使用模型" in text for text in result.feedback))

    def test_no_service_has_an_actionable_chinese_error_without_retry(self):
        self.stop_server()
        adapter = OllamaDecomposer(host=unused_endpoint(), timeout=1)
        with self.assertRaisesRegex(OllamaError, "未检测到本机 Ollama 服务，请先启动 Ollama"):
            adapter.decompose(self.problem)
        self.assertEqual(len(adapter.calls), 1)
        self.assertFalse(adapter.calls[0]["json_valid"])

    def test_no_service_falls_back_without_crashing_the_pipeline(self):
        self.problem.proposal = None
        baseline = run(self.problem)
        self.stop_server()
        result = run(self.problem, decomposer=OllamaDecomposer(host=unused_endpoint(), timeout=1))
        self.assertEqual(result.status, "solved")
        self.assertIsNone(result.proposal)
        self.assertIsNone(result.completion)
        self.assertEqual(len(result.model_calls), 1)
        self.assertIn("未检测到本机 Ollama 服务，请先启动 Ollama", result.model_error)
        self.assertLessEqual(abs(result.schedule.makespan - baseline.schedule.makespan), 1e-6)

    def test_redirect_is_rejected_instead_of_contacting_another_host(self):
        self.server.responses.append((302, {"Location": "http://example.invalid/"}, {}))
        adapter = OllamaDecomposer(host=self.host)
        with self.assertRaisesRegex(OllamaError, "重定向|HTTP 302"):
            adapter.decompose(self.problem)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(len(adapter.calls), 1)

    def test_http_and_api_errors_are_recorded_without_parse_retries(self):
        for status, text in ((404, "模型不存在"), (200, "模型加载失败")):
            with self.subTest(status=status):
                self.server.responses.append((status, {}, {"error": text}))
                adapter = OllamaDecomposer(host=self.host)
                with self.assertRaisesRegex(OllamaError, text):
                    adapter.decompose(self.problem)
                self.assertEqual(len(adapter.calls), 1)
                self.assertIn(text, adapter.calls[0]["raw_api_response"])

    def test_mode_and_deterministic_options_can_be_configured(self):
        self.server.responses.append(response())
        adapter = OllamaDecomposer(host=self.host, mode="minimal", temperature=0.25, seed=42, timeout=10)
        adapter.decompose(self.problem)
        payload = self.server.requests[0][1]
        self.assertNotIn("能力摘要", payload["prompt"])
        self.assertEqual(payload["options"]["temperature"], 0.25)
        self.assertEqual(payload["options"]["seed"], 42)
        self.assertEqual(adapter.calls[0]["mode"], "minimal")


class OllamaConfigurationTests(unittest.TestCase):
    def test_host_is_limited_to_explicit_loopback_endpoints(self):
        for host in ("https://127.0.0.1:12345", "http://example.invalid:12345",
                     "http://user:secret@127.0.0.1:12345", "http://127.0.0.1:12345/path"):
            with self.subTest(host=host):
                with self.assertRaisesRegex(OllamaError, "本机|地址|127.0.0.1"):
                    OllamaDecomposer(host=host)

    def test_loopback_port_selection_is_preserved_without_network_access(self):
        for host in ("http://127.0.0.1:12345", "http://localhost:12345"):
            with self.subTest(host=host):
                self.assertEqual(OllamaDecomposer(host=host).host, host)

    def test_configuration_rejects_invalid_timeout_mode_or_retry_count(self):
        for options in ({"timeout": 0}, {"timeout": float("nan")}, {"mode": "other"},
                        {"max_retries": 3}, {"max_retries": -1}, {"model": ""}):
            with self.subTest(options=options):
                with self.assertRaises(OllamaError):
                    OllamaDecomposer(**options)


if __name__ == "__main__":
    unittest.main()
