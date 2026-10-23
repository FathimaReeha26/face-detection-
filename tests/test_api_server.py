import email.parser
import email.policy
import unittest

import api_server


class TestApiServer(unittest.TestCase):
    def test_auth_no_key_required_by_default(self):
        # When API_KEY is empty, check_auth should return True
        orig_key = api_server.API_KEY
        try:
            api_server.API_KEY = ""
            self.assertTrue(api_server.check_auth({}))
        finally:
            api_server.API_KEY = orig_key

    def test_auth_with_key(self):
        orig_key = api_server.API_KEY
        try:
            api_server.API_KEY = "secret_key_123"
            self.assertTrue(api_server.check_auth({"X-API-Key": "secret_key_123"}))
            self.assertFalse(api_server.check_auth({"X-API-Key": "wrong_key"}))
            self.assertFalse(api_server.check_auth({}))
        finally:
            api_server.API_KEY = orig_key

    def test_b64_decode_valid_and_invalid(self):
        import base64
        valid_b64 = base64.b64encode(b"hello world").decode("ascii")
        self.assertEqual(base64.b64decode(valid_b64, validate=True), b"hello world")

        with self.assertRaises(Exception):
            base64.b64decode("not-valid-base-64!!!", validate=True)


if __name__ == "__main__":
    unittest.main()
