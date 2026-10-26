"""Run the new TestSequenceFileTypeValidation tests without full deepchem import.

The repo test file does `from deepchem.utils import sequence_utils`, which
triggers deepchem/__init__ -> rdkit (not installed here). So stub the package
path and load both modules by file location, then run unittest.
"""
import importlib.util
import pathlib
import sys
import types
import unittest

REPO = pathlib.Path(r"E:\Bdtorrent\CHikni\work\deepchem")
SU_PATH = REPO / "deepchem" / "utils" / "sequence_utils.py"
TEST_PATH = REPO / "deepchem" / "utils" / "test" / "test_sequence_utils.py"

# Build stub package hierarchy: deepchem -> deepchem.utils -> sequence_utils
su_spec = importlib.util.spec_from_file_location(
    "deepchem.utils.sequence_utils", str(SU_PATH))
su_mod = importlib.util.module_from_spec(su_spec)
pkg = types.ModuleType("deepchem")
pkg.__path__ = [str(REPO / "deepchem")]
utils_pkg = types.ModuleType("deepchem.utils")
utils_pkg.__path__ = [str(REPO / "deepchem" / "utils")]
sys.modules["deepchem"] = pkg
sys.modules["deepchem.utils"] = utils_pkg
sys.modules["deepchem.utils.sequence_utils"] = su_mod
pkg.utils = utils_pkg
utils_pkg.sequence_utils = su_mod
su_spec.loader.exec_module(su_mod)

test_spec = importlib.util.spec_from_file_location(
    "test_sequence_utils_new", str(TEST_PATH))
test_mod = importlib.util.module_from_spec(test_spec)
test_spec.loader.exec_module(test_mod)

suite = unittest.TestLoader().loadTestsFromTestCase(
    test_mod.TestSequenceFileTypeValidation)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
