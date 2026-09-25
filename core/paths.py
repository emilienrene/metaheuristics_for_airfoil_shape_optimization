"""
Path resolution for the optimization run hierarchy.

All scripts import from here instead of each duplicating glob logic.
"""

from __future__ import annotations

import glob
import os
import re


def repo_root() -> str:
    """Absolute path of the repository root (directory containing this file's parent)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def latest_opt_dir() -> str:
    env_opt_dir = os.environ.get("OPT_DIR")
    if env_opt_dir:
        return os.path.abspath(env_opt_dir)

    root = repo_root()
    dirs = glob.glob(os.path.join(root, "optimization_*"))
    if not dirs:
        raise RuntimeError("No optimization_* directories found.")
    return max(dirs, key=lambda p: int(re.search(r"optimization_(\d+)", p).group(1)))


def design_space_dir(opt_dir: str | None = None) -> str:
    return os.path.join(opt_dir or latest_opt_dir(), "Design_space")


def latest_eval_dir(opt_dir: str | None = None, pattern: str = "evaluation_mach_*") -> str:
    opt = opt_dir or latest_opt_dir()
    dirs = sorted(glob.glob(os.path.join(opt, pattern)))
    if not dirs:
        raise RuntimeError(f"No {pattern} directory found in {opt}")
    return dirs[-1]


def clmax_eval_dir(opt_dir: str | None = None) -> str:
    opt = opt_dir or latest_opt_dir()
    dirs = sorted(glob.glob(os.path.join(opt, "evaluation_Mach_*")))
    if not dirs:
        raise RuntimeError(f"No evaluation_Mach_* directory found in {opt}")
    return dirs[-1]


def next_gen_index(opt_dir: str | None = None) -> int:
    opt = opt_dir or latest_opt_dir()
    dirs = glob.glob(os.path.join(opt, "generation_*"))
    indices = [
        int(os.path.basename(d).split("_")[-1])
        for d in dirs
        if os.path.basename(d).split("_")[-1].isdigit()
    ]
    return 1 if not indices else max(indices) + 1


def generation_dir(index: int, opt_dir: str | None = None) -> str:
    return os.path.join(opt_dir or latest_opt_dir(), f"generation_{index}")
