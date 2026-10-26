"""Reproduce deepchem issue #5186 without importing full deepchem (avoids rdkit).

Mirrors the issue's 3 steps: unsupported ext accepted, -M first missing,
logging.raiseExceptions is a bool.
"""
import importlib.util
import os
import pathlib
import tempfile

MOD = pathlib.Path(r"E:\Bdtorrent\CHikni\work\deepchem\deepchem\utils\sequence_utils.py")
spec = importlib.util.spec_from_file_location("su_under_test", str(MOD))
su = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su)

captured = []
su.system_call = lambda cmd: captured.append(cmd)

tmp = tempfile.mkdtemp()


def query(name):
    p = os.path.join(tmp, name)
    with open(p, "w") as f:
        f.write(">s\nMKV\n")
    return p


print("=== 1. unsupported extension (.txt) should raise, currently does not ===")
try:
    out = su.hhsearch(query("seq.txt"), database="db", data_dir=tmp)
    print(f"BUG: no error raised, returned {out}")
    print(f"BUG: command built was: {captured[-1]}")
except Exception as e:
    print(f"OK raised {type(e).__name__}: {e}")

print()
print("=== 2. -M first flag: .fasta vs .a3m should differ ===")
for ext in (".fasta", ".a3m"):
    captured.clear()
    su.hhsearch(query("q" + ext), database="db", data_dir=tmp)
    print(f"{ext}: {captured[-1]}")
    print(f"  has '-M first': {'-M first' in captured[-1]}")

print()
print("=== 3. raiseExceptions is a bool, not a function ===")
from logging import raiseExceptions
print(f"type={type(raiseExceptions).__name__} value={raiseExceptions!r}")
try:
    raiseExceptions("boom")
    print("BUG: callable (unexpected)")
except TypeError as e:
    print(f"BUG confirmed: calling it gives TypeError: {e}")
