import os
import unittest

from detection.config import (
    DEFAULT_MODELS_DIR,
    MODE_VARIABLE,
    MODEL_VARIABLE,
    DetectorSettings,
    BACKEND_CONFIGS,
)


class TestConfig(unittest.TestCase):
    def test_default_settings(self):
        settings = DetectorSettings()
        self.assertEqual(settings.mode, "auto")
        self.assertEqual(settings.model, "auto")
        self.assertEqual(settings.models_dir, DEFAULT_MODELS_DIR)

    def test_from_env_defaults(self):
        # Clear env vars if present
        for var in (MODE_VARIABLE, MODEL_VARIABLE):
            os.environ.pop(var, None)

        settings = DetectorSettings.from_env()
        self.assertEqual(settings.mode, "auto")
        self.assertEqual(settings.model, "auto")

    def test_from_env_valid_override(self):
        os.environ[MODE_VARIABLE] = "cpu"
        os.environ[MODEL_VARIABLE] = "yunet"
        try:
            settings = DetectorSettings.from_env()
            self.assertEqual(settings.mode, "cpu")
            self.assertEqual(settings.model, "yunet")
        finally:
            os.environ.pop(MODE_VARIABLE, None)
            os.environ.pop(MODEL_VARIABLE, None)

    def test_from_env_invalid_override(self):
        os.environ[MODE_VARIABLE] = "quantum_gpu"
        try:
            settings = DetectorSettings.from_env()
            self.assertEqual(settings.mode, "auto")
        finally:
            os.environ.pop(MODE_VARIABLE, None)

    def test_backend_configs_presence(self):
        self.assertIn("yunet", BACKEND_CONFIGS)
        self.assertIn("scrfd", BACKEND_CONFIGS)
        self.assertIn("retinaface", BACKEND_CONFIGS)
        for name, cfg in BACKEND_CONFIGS.items():
            self.assertGreater(cfg.score_threshold, 0.0)
            self.assertGreater(cfg.nms_threshold, 0.0)
            self.assertEqual(len(cfg.input_size), 2)


if __name__ == "__main__":
    unittest.main()
