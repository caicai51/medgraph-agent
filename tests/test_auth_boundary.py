import os
import hashlib
import hmac
import unittest
from types import SimpleNamespace

from app.streaming_api import resolve_user_id_from_request


class AuthBoundaryTests(unittest.TestCase):
    def test_accepts_signed_proxy_identity_over_body_value(self):
        os.environ["AUTH_PROXY_SHARED_SECRET"] = "unit-test-secret"
        signature = hmac.new(b"unit-test-secret", b"alice", hashlib.sha256).hexdigest()
        request = SimpleNamespace(headers={"X-Authenticated-User-Id": "alice", "X-Authenticated-User-Signature": signature})
        self.assertEqual(resolve_user_id_from_request(request, body_user_id="mallory"), "alice")
        os.environ.pop("AUTH_PROXY_SHARED_SECRET", None)

    def test_rejects_unsigned_identity_header(self):
        request = SimpleNamespace(headers={"X-Authenticated-User-Id": "mallory"})
        self.assertIsNone(resolve_user_id_from_request(request))

    def test_rejects_body_user_id_without_trusted_header(self):
        request = SimpleNamespace(headers={})
        self.assertIsNone(resolve_user_id_from_request(request, body_user_id="mallory"))

    def test_explicit_untrusted_fallback_only_when_enabled(self):
        os.environ["ALLOW_UNTRUSTED_USER_ID"] = "true"
        try:
            request = SimpleNamespace(headers={})
            self.assertEqual(resolve_user_id_from_request(request, body_user_id="mallory"), "mallory")
        finally:
            os.environ.pop("ALLOW_UNTRUSTED_USER_ID", None)

    def test_demo_header_is_only_accepted_when_dev_mode_enabled(self):
        request = SimpleNamespace(headers={"X-User-Id": "demo-user"})
        self.assertIsNone(resolve_user_id_from_request(request))
        os.environ["ALLOW_UNTRUSTED_USER_ID"] = "true"
        try:
            self.assertEqual(resolve_user_id_from_request(request), "demo-user")
        finally:
            os.environ.pop("ALLOW_UNTRUSTED_USER_ID", None)


if __name__ == "__main__":
    unittest.main()
