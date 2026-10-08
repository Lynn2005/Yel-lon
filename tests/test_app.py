"""Regression tests for Lynn Recap's pure helpers and API-key persistence wiring.

Run locally with: python -m unittest discover -s tests -v
These tests avoid importing app.py because Streamlit executes the UI at import time.
"""
import ast
import re
import unittest
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_SOURCE = APP_PATH.read_text(encoding="utf-8")
TREE = ast.parse(APP_SOURCE)


def load_function(name, namespace):
    node = next(
        item for item in TREE.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace[name]


class HelperFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = {"re": re, "Path": Path}
        cls.srt_time = load_function("srt_time", ns)
        cls.make_srt = load_function("make_srt", ns)
        cls.safe_name = load_function("safe_name", ns)

    def test_srt_time_formats_milliseconds(self):
        self.assertEqual(self.srt_time(65.432), "00:01:05,432")

    def test_make_srt_creates_valid_numbered_entries(self):
        actual = self.make_srt([
            {"start": 0, "end": 1.25, "text": "Hello"},
            {"start": 1.0, "end": 2.5, "text": "World"},
        ])
        self.assertIn("1\n00:00:00,000 --> 00:00:01,250\nHello", actual)
        self.assertIn("2\n00:00:01,250 --> 00:00:02,500\nWorld", actual)

    def test_make_srt_skips_empty_text(self):
        actual = self.make_srt([{"start": 0, "end": 1, "text": "  "}])
        self.assertEqual(actual, "\n")

    def test_safe_name_removes_path_and_unsafe_characters(self):
        self.assertEqual(self.safe_name("../../my movie!.mp4"), "my_movie_.mp4")


class ApiKeyPersistenceRegressionTests(unittest.TestCase):

    def test_gemini_uses_current_model_ids_not_shutdown_gemini_2(self):
        self.assertIn('model="gemini-3.8-flash"', APP_SOURCE)
        self.assertIn('"gemini-3.7-flash"', APP_SOURCE)
        self.assertIn('"gemini-3.5-flash-lite"', APP_SOURCE)
        self.assertNotIn("gemini-2.0-flash", APP_SOURCE)
        self.assertNotIn("gemini-2.5-flash", APP_SOURCE)
    def test_reads_use_supported_getitem_signature(self):
        self.assertIn('local_storage.getItem("yel_lon_groq_api_key")', APP_SOURCE)
        self.assertIn('local_storage.getItem("yel_lon_gemini_api_key")', APP_SOURCE)
        self.assertNotIn('getItem("yel_lon_groq_api_key", key=', APP_SOURCE)
        self.assertNotIn('getItem("yel_lon_gemini_api_key", key=', APP_SOURCE)

    def test_groq_and_gemini_writes_have_distinct_component_keys(self):
        self.assertIn(
            'local_storage.setItem("yel_lon_groq_api_key", groq_to_save, key="save_groq_api_key")',
            APP_SOURCE,
        )
        self.assertIn(
            'local_storage.setItem("yel_lon_gemini_api_key", gemini_to_save, key="save_gemini_api_key")',
            APP_SOURCE,
        )
        self.assertNotEqual("save_groq_api_key", "save_gemini_api_key")

    def test_saved_values_are_reused_after_rerun(self):
        self.assertIn('st.session_state.saved_groq_key = stored_groq or os.getenv("GROQ_API_KEY", "")', APP_SOURCE)
        self.assertIn('st.session_state.saved_gemini_key = stored_gemini or os.getenv("GEMINI_API_KEY", "")', APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
