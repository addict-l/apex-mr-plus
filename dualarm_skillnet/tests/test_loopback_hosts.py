"""Loopback configuration is independent of the real Ollama port."""

import socket
import unittest

from skillnet.decompose.ollama import DEFAULT_HOST, OllamaDecomposer, OllamaError


def vacant_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class LoopbackAddressTests(unittest.TestCase):
    def test_accepts_explicit_loopback_ports_and_keeps_selected_address(self):
        port = vacant_port()
        for host in (f"http://127.0.0.1:{port}", f"http://localhost:{port}",
                     "http://127.0.0.1:1", "http://localhost:65535"):
            with self.subTest(host=host):
                adapter = OllamaDecomposer(host=host)
                self.assertEqual(adapter.host, host)
        self.assertEqual(OllamaDecomposer().host, DEFAULT_HOST)

    def test_rejects_non_loopback_hosts_credentials_paths_and_invalid_ports(self):
        for host in (
            "http://example.invalid:12345", "http://127.0.0.2:12345",
            "http://localhost.example.invalid:12345", "http://[::1]:12345",
            "https://127.0.0.1:12345", "http://user:pass@localhost:12345",
            "http://@localhost:12345", "http://localhost:12345/", "http://localhost:12345/api",
            "http://localhost:12345?query=1", "http://localhost:12345#fragment",
            "http://127.0.0.1", "http://localhost:0", "http://localhost:65536",
            "http://localhost:bad", "localhost:12345",
        ):
            with self.subTest(host=host), self.assertRaisesRegex(OllamaError, "地址|本机|端口"):
                OllamaDecomposer(host=host)

    def test_natural_language_mode_is_a_supported_adapter_mode(self):
        adapter = OllamaDecomposer(mode="nl_only", host=f"http://127.0.0.1:{vacant_port()}")
        self.assertEqual(adapter.mode, "nl_only")


if __name__ == "__main__":
    unittest.main()
