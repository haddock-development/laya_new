"""Exercise secret-file handling across the container command boundary."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker" / "entrypoint.py"


def load_entrypoint_module():
    spec = importlib.util.spec_from_file_location("laya_docker_entrypoint", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EntrypointTests(unittest.TestCase):
    def invoke(self, extra, expected="direct"):
        env = {key: value for key, value in os.environ.items()
               if key not in {"HF_TOKEN", "HF_TOKEN_FILE"}}
        env.update(extra)
        child = "import os, sys; assert os.environ.get('HF_TOKEN') == sys.argv[1]; print('child ran')"
        return subprocess.run(
            [sys.executable, str(ENTRYPOINT), sys.executable, "-c", child, expected],
            env=env, capture_output=True, text=True, check=False,
        )

    def test_direct_value(self):
        result = self.invoke({"HF_TOKEN": "direct"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "child ran")

    def test_file_overrides_direct_and_trims_whitespace(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "token"
            secret.write_text("  synthetic-file-value\n", encoding="utf-8")
            result = self.invoke({"HF_TOKEN": "direct", "HF_TOKEN_FILE": str(secret)},
                                 "synthetic-file-value")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("synthetic-file-value", result.stdout + result.stderr)

    def test_invalid_files_stop_before_child_without_disclosure(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "private-filename"
            for contents in (None, b"", b" \n", b"synthetic-secret\x00", b"\xff"):
                with self.subTest(contents=contents):
                    if contents is not None:
                        secret.write_bytes(contents)
                    result = self.invoke({"HF_TOKEN": "direct", "HF_TOKEN_FILE": str(secret)})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("HF_TOKEN_FILE", result.stderr)
                    self.assertNotIn(str(secret), result.stderr)
                    self.assertNotIn("synthetic-secret", result.stderr)
                    self.assertNotIn("direct", result.stderr)

    def test_empty_file_setting_and_unknown_suffix_are_ignored(self):
        result = self.invoke({"HF_TOKEN": "direct", "HF_TOKEN_FILE": "", "UNKNOWN_FILE": "/missing"})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_command_fails(self):
        result = subprocess.run([sys.executable, str(ENTRYPOINT)], env={},
                                capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("command is required", result.stderr)

    def test_root_startup_drops_to_the_dedicated_laya_account(self):
        entrypoint = load_entrypoint_module()
        with mock.patch.object(entrypoint.os, "geteuid", return_value=0), \
             mock.patch.object(entrypoint.os, "setgroups") as setgroups, \
             mock.patch.object(entrypoint.os, "setgid") as setgid, \
             mock.patch.object(entrypoint.os, "setuid") as setuid:
            entrypoint.drop_privileges_if_root()

        setgroups.assert_called_once_with([])
        setgid.assert_called_once_with(10001)
        setuid.assert_called_once_with(10001)

    def test_non_root_startup_keeps_its_current_identity(self):
        entrypoint = load_entrypoint_module()
        with mock.patch.object(entrypoint.os, "geteuid", return_value=10001), \
             mock.patch.object(entrypoint.os, "setgroups") as setgroups, \
             mock.patch.object(entrypoint.os, "setgid") as setgid, \
             mock.patch.object(entrypoint.os, "setuid") as setuid:
            entrypoint.drop_privileges_if_root()

        setgroups.assert_not_called()
        setgid.assert_not_called()
        setuid.assert_not_called()


if __name__ == "__main__":
    unittest.main()
