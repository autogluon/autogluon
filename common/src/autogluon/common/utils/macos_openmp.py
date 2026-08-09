"""
macOS OpenMP compatibility for pip installs of torch + lightgbm.

Background
----------
On macOS, multiple packages may each resolve a different ``libomp.dylib``:

* **torch** vendors ``site-packages/torch/lib/libomp.dylib``
* **lightgbm** uses ``@rpath/libomp.dylib`` with Homebrew rpaths
* **scikit-learn** vendors ``site-packages/sklearn/.dylibs/libomp.dylib``

Loading more than one OpenMP runtime in a process can SIGSEGV under multi-threaded
use (pytorch#191933, LightGBM#6595, AutoGluon#5793).

Strategy (packaging-native, not absolute path hacks)
----------------------------------------------------
1. **lightgbm**: keep dependency ``@rpath/libomp.dylib``; replace system rpaths with a
   single relative rpath ``@loader_path/../../torch/lib`` so dyld finds torch's libomp
   next to both packages under ``site-packages``.
2. **scikit-learn**: replace the vendored ``.dylibs/libomp.dylib`` with a **symlink** to
   torch's libomp so existing ``@loader_path/.../libomp.dylib`` references resolve to
   the same file.

This is the same resolution model wheels already use (rpath / loader-relative paths),
applied once per environment. It is idempotent and survives relocating a whole
``site-packages`` tree when torch and lightgbm stay siblings.

Auto vs CLI
-----------
* ``ensure_fixed()`` — called from ``try_import_lightgbm`` / ``try_import_torch`` on Darwin;
  never raises; lightgbm rpath + sklearn symlink; once per process.
* CLI ``fix`` / ``check`` / ``smoke`` — same operations with fuller logging for support/CI.

Disable auto-fix: ``AUTOGLUON_DISABLE_MACOS_OPENMP_AUTOFIX=1``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ENV_DISABLE_AUTOFIX = "AUTOGLUON_DISABLE_MACOS_OPENMP_AUTOFIX"

# System rpaths commonly baked into lightgbm macOS wheels
_SYSTEM_LIBOMP_RPATHS = frozenset(
    {
        "/opt/homebrew/opt/libomp/lib",
        "/opt/local/lib/libomp",
        "/usr/local/opt/libomp/lib",
    }
)

_OTOOL_DEP_RE = re.compile(r"^\t(.+libomp\.dylib)(?:\s+\(|$)")
_RPATH_PREFIX = "@loader_path/"

EnsureFixedStatus = Literal["skipped", "ok", "fixed", "failed"]

# Process-local guard (import hooks call this frequently).
_ensure_state: EnsureFixedStatus | None = None


@dataclass(frozen=True)
class CompatState:
    """Record of the last successful environment alignment."""

    torch_libomp: str
    lightgbm_rpath: str | None
    sklearn_symlink: str | None
    applied_unix: float

    def to_json(self) -> dict:
        return {
            "torch_libomp": self.torch_libomp,
            "lightgbm_rpath": self.lightgbm_rpath,
            "sklearn_symlink": self.sklearn_symlink,
            "applied_unix": self.applied_unix,
            "version": 1,
        }

    @classmethod
    def from_json(cls, data: dict) -> "CompatState":
        return cls(
            torch_libomp=data["torch_libomp"],
            lightgbm_rpath=data.get("lightgbm_rpath"),
            sklearn_symlink=data.get("sklearn_symlink"),
            applied_unix=float(data.get("applied_unix", 0)),
        )


def is_macos() -> bool:
    return sys.platform == "darwin"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError(f"{tool!r} not found on PATH. Install Xcode Command Line Tools (`xcode-select --install`).")
    return path


def _autofix_disabled() -> bool:
    return os.environ.get(ENV_DISABLE_AUTOFIX, "").strip().lower() in {"1", "true", "yes", "on"}


def get_torch_lib_dir() -> Path:
    """Return ``.../site-packages/torch/lib`` (must contain ``libomp.dylib``).

    Prefer filesystem discovery so we do not load torch's libomp before alignment.
    Falls back to ``import torch`` if needed.
    """
    for root in _site_package_roots():
        lib_dir = root / "torch" / "lib"
        if (lib_dir / "libomp.dylib").is_file():
            return lib_dir.resolve()
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("torch is not installed.") from e
    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if not (lib_dir / "libomp.dylib").is_file():
        raise RuntimeError(
            f"torch does not vendor libomp at {lib_dir / 'libomp.dylib'}; cannot align OpenMP load paths."
        )
    return lib_dir


def get_canonical_libomp() -> Path:
    """Absolute path to torch's vendored ``libomp.dylib``."""
    return get_torch_lib_dir() / "libomp.dylib"


def _site_package_roots() -> list[Path]:
    """Candidate site-packages roots without importing third-party packages."""
    roots: list[Path] = []
    try:
        import site

        for p in list(site.getsitepackages()) + [site.getusersitepackages()]:
            if p:
                roots.append(Path(p))
    except Exception:
        pass
    for p in sys.path:
        if not p:
            continue
        path = Path(p)
        # editable installs: .../common/src
        if path.name == "src" or path.name == "site-packages" or "site-packages" in path.parts:
            roots.append(path)
        roots.append(path)
    # de-dupe preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        try:
            rp = r.resolve()
        except OSError:
            rp = r
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def _find_lightgbm_dylib() -> Path | None:
    """Locate lib_lightgbm without importing lightgbm (import loads OpenMP too early)."""
    for root in _site_package_roots():
        matches = sorted(root.glob("lightgbm/**/lib_lightgbm*.dylib"))
        if not matches:
            matches = sorted(root.glob("**/lightgbm/**/lib_lightgbm*.dylib"))
        if matches:
            return matches[0]
    return None


def _sklearn_vendored_libomp() -> Path | None:
    """Locate sklearn's vendored libomp without importing sklearn."""
    for root in _site_package_roots():
        path = root / "sklearn" / ".dylibs" / "libomp.dylib"
        if path.exists() or path.is_symlink() or path.parent.is_dir():
            return path
        # nested layouts
        matches = list(root.glob("**/sklearn/.dylibs/libomp.dylib"))
        if matches:
            return matches[0]
    return None


def _otool_libomp_deps(binary: Path) -> list[str]:
    _require("otool")
    result = _run(["otool", "-L", str(binary)])
    if result.returncode != 0:
        return []
    deps: list[str] = []
    for line in result.stdout.splitlines():
        m = _OTOOL_DEP_RE.match(line)
        if m:
            deps.append(m.group(1))
    return deps


def _otool_rpaths(binary: Path) -> list[str]:
    _require("otool")
    result = _run(["otool", "-l", str(binary)])
    if result.returncode != 0:
        return []
    rpaths: list[str] = []
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if "cmd LC_RPATH" not in line:
            continue
        for j in range(i + 1, min(i + 6, len(lines))):
            if "path " not in lines[j]:
                continue
            part = lines[j].strip().split("path ", 1)[-1]
            rpaths.append(part.split(" (offset", 1)[0].strip())
            break
    return rpaths


def _loader_relative_rpath(binary: Path, target_dir: Path) -> str:
    """Rpath from ``binary``'s directory to ``target_dir`` using ``@loader_path``."""
    rel = os.path.relpath(target_dir.resolve(), start=binary.parent.resolve())
    # normalize to forward slashes for dyld
    rel = rel.replace(os.sep, "/")
    return f"{_RPATH_PREFIX}{rel}"


def _codesign(binary: Path) -> None:
    """Re-sign after install_name_tool (required on modern macOS)."""
    codesign = shutil.which("codesign")
    if codesign is None:
        return
    _run([codesign, "-s", "-", "-f", str(binary)])


def _state_path(torch_lib: Path) -> Path:
    # Store next to torch so it is env-local (venv/conda prefix).
    return torch_lib.parent / ".autogluon_macos_openmp.json"


def _read_state(torch_lib: Path) -> CompatState | None:
    path = _state_path(torch_lib)
    try:
        data = json.loads(path.read_text())
        return CompatState.from_json(data)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _write_state(state: CompatState, torch_lib: Path) -> None:
    path = _state_path(torch_lib)
    try:
        path.write_text(json.dumps(state.to_json(), indent=2) + "\n")
    except OSError as e:
        logger.log(10, f"Could not write OpenMP compat state file {path}: {e}")


def _lightgbm_desired_rpath(lgb: Path, torch_lib: Path) -> str:
    return _loader_relative_rpath(lgb, torch_lib)


def lightgbm_is_aligned(torch_lib: Path | None = None) -> bool:
    """True if lightgbm is missing or already resolves libomp via torch's lib dir."""
    lgb = _find_lightgbm_dylib()
    if lgb is None:
        return True
    torch_lib = torch_lib or get_torch_lib_dir()
    desired = _lightgbm_desired_rpath(lgb, torch_lib)
    rpaths = _otool_rpaths(lgb)
    deps = _otool_libomp_deps(lgb)
    # Accept either relative rpath layout or absolute link to torch libomp (legacy).
    canonical = torch_lib / "libomp.dylib"
    if any(d.startswith("/") and Path(d).resolve() == canonical.resolve() for d in deps):
        return not any(r in _SYSTEM_LIBOMP_RPATHS for r in rpaths)
    if not any(d == "@rpath/libomp.dylib" or d.endswith("libomp.dylib") for d in deps):
        return True
    if desired not in rpaths:
        return False
    if any(r in _SYSTEM_LIBOMP_RPATHS for r in rpaths):
        return False
    return True


def sklearn_is_aligned(torch_lib: Path | None = None) -> bool:
    """True if sklearn is missing or its vendored libomp is a symlink to torch's."""
    sk = _sklearn_vendored_libomp()
    if sk is None or not sk.exists():
        return True
    torch_lib = torch_lib or get_torch_lib_dir()
    canonical = (torch_lib / "libomp.dylib").resolve()
    try:
        return sk.is_symlink() and sk.resolve() == canonical
    except OSError:
        return False


def needs_fix() -> bool:
    if not is_macos():
        return False
    try:
        torch_lib = get_torch_lib_dir()
    except RuntimeError:
        return False
    return not (lightgbm_is_aligned(torch_lib) and sklearn_is_aligned(torch_lib))


def _align_lightgbm(torch_lib: Path, *, dry_run: bool) -> str | None:
    """Align lightgbm rpaths. Returns the desired rpath string, or None if lightgbm absent."""
    lgb = _find_lightgbm_dylib()
    if lgb is None:
        return None

    desired = _lightgbm_desired_rpath(lgb, torch_lib)
    current_rpaths = _otool_rpaths(lgb)
    deps = _otool_libomp_deps(lgb)

    if dry_run:
        logger.log(20, f"[dry-run] lightgbm rpaths {current_rpaths} -> [{desired}]")
        return desired

    if not os.access(lgb, os.W_OK):
        raise RuntimeError(f"No write permission for {lgb}")

    _require("install_name_tool")

    # Prefer standard @rpath/libomp.dylib dependency (wheel default).
    for dep in deps:
        if dep != "@rpath/libomp.dylib" and "libomp.dylib" in dep:
            result = _run(["install_name_tool", "-change", dep, "@rpath/libomp.dylib", str(lgb)])
            if result.returncode != 0:
                logger.log(10, f"install_name_tool -change dep failed: {result.stderr}")

    # Drop system / stale rpaths, then ensure desired relative rpath exists.
    for rpath in current_rpaths:
        if rpath == desired:
            continue
        _run(["install_name_tool", "-delete_rpath", rpath, str(lgb)])

    if desired not in _otool_rpaths(lgb):
        result = _run(["install_name_tool", "-add_rpath", desired, str(lgb)])
        if result.returncode != 0:
            raise RuntimeError(
                f"install_name_tool -add_rpath failed for {lgb}: {(result.stderr or result.stdout).strip()}"
            )

    _codesign(lgb)
    return desired


def _align_sklearn(torch_lib: Path, *, dry_run: bool) -> str | None:
    """Point sklearn's vendored libomp at torch's via symlink. Returns target path or None."""
    sk = _sklearn_vendored_libomp()
    if sk is None:
        return None
    # If sklearn has no .dylibs dir yet / never shipped omp, skip
    dylibs_dir = sk.parent
    if not dylibs_dir.is_dir() and not sk.exists():
        return None

    canonical = torch_lib / "libomp.dylib"
    if dry_run:
        logger.log(20, f"[dry-run] symlink {sk} -> {canonical}")
        return str(canonical)

    if sk.exists() or sk.is_symlink():
        if sk.is_symlink() and sk.resolve() == canonical.resolve():
            return str(canonical)
        if not os.access(sk.parent, os.W_OK):
            raise RuntimeError(f"No write permission for {sk.parent}")
        sk.unlink()

    if not dylibs_dir.is_dir():
        return None

    sk.symlink_to(canonical)
    return str(canonical)


def fix(*, dry_run: bool = False) -> int:
    """
    Align lightgbm + scikit-learn OpenMP resolution to torch's libomp.

    Returns
    -------
    int
        0 success, 1 soft failure, 2 hard error.
    """
    if not is_macos():
        logger.log(20, "Not macOS; OpenMP compatibility fix is a no-op.")
        return 0

    try:
        torch_lib = get_torch_lib_dir()
    except RuntimeError as e:
        logger.log(40, str(e))
        return 2

    logger.log(20, f"Aligning OpenMP load paths to {torch_lib / 'libomp.dylib'}")

    try:
        lgb_rpath = _align_lightgbm(torch_lib, dry_run=dry_run)
        sk_link = _align_sklearn(torch_lib, dry_run=dry_run)
    except Exception as e:
        logger.log(40, f"OpenMP compatibility fix failed: {type(e).__name__}: {e}")
        return 2

    if dry_run:
        logger.log(20, "Dry run complete; no files modified.")
        return 0

    _write_state(
        CompatState(
            torch_libomp=str(torch_lib / "libomp.dylib"),
            lightgbm_rpath=lgb_rpath,
            sklearn_symlink=sk_link,
            applied_unix=time.time(),
        ),
        torch_lib,
    )
    # Invalidate process cache so subsequent ensure_fixed re-checks if needed
    global _ensure_state
    _ensure_state = "ok"
    logger.log(20, "OpenMP compatibility fix applied.")
    return 0


def check() -> int:
    """
    Verify alignment. Exit 0 if OK, 1 if not aligned, 2 on hard error.
    """
    if not is_macos():
        logger.log(20, "Not macOS; OpenMP check skipped.")
        return 0

    try:
        torch_lib = get_torch_lib_dir()
    except RuntimeError as e:
        logger.log(40, str(e))
        return 2

    canonical = torch_lib / "libomp.dylib"
    problems: list[str] = []

    lgb = _find_lightgbm_dylib()
    if lgb is None:
        logger.log(20, "lightgbm not installed; lightgbm check skipped.")
    else:
        deps = _otool_libomp_deps(lgb)
        rpaths = _otool_rpaths(lgb)
        logger.log(20, f"lightgbm: {lgb}")
        logger.log(20, f"\tdeps={deps}")
        logger.log(20, f"\trpaths={rpaths}")
        if not lightgbm_is_aligned(torch_lib):
            problems.append(
                "lightgbm is not aligned to torch's libomp. "
                "AutoGluon will attempt to fix this on import, or run: "
                "python -m autogluon.common.utils.macos_openmp fix"
            )

    sk = _sklearn_vendored_libomp()
    if sk is not None and sk.parent.is_dir():
        logger.log(20, f"sklearn vendored libomp: {sk} (symlink={sk.is_symlink()})")
        if sk.exists() and not sklearn_is_aligned(torch_lib):
            problems.append(f"sklearn libomp is not a symlink to {canonical}")
    else:
        logger.log(20, "scikit-learn vendored libomp not present; sklearn check skipped.")

    if problems:
        for msg in problems:
            logger.log(30, msg)
        return 1

    logger.log(20, f"OpenMP check OK (canonical={canonical}).")
    return 0


def ensure_fixed(*, force: bool = False) -> EnsureFixedStatus:
    """
    Idempotent OpenMP alignment for macOS. Safe for import hooks; never raises.

    Prefer calling only via ``try_import_lightgbm`` / ``try_import_torch`` so the
    fix runs before native libraries are loaded.
    """
    global _ensure_state

    if _ensure_state is not None and not force:
        return _ensure_state

    if not is_macos() or _autofix_disabled():
        _ensure_state = "skipped"
        return _ensure_state

    try:
        torch_lib = get_torch_lib_dir()
    except RuntimeError:
        _ensure_state = "skipped"
        return _ensure_state

    try:
        if lightgbm_is_aligned(torch_lib) and sklearn_is_aligned(torch_lib):
            _ensure_state = "ok"
            return _ensure_state
    except Exception as e:
        logger.log(10, f"OpenMP alignment check failed: {e}")
        _ensure_state = "failed"
        return _ensure_state

    try:
        lgb_rpath = _align_lightgbm(torch_lib, dry_run=False)
        sk_link = _align_sklearn(torch_lib, dry_run=False)
        _write_state(
            CompatState(
                torch_libomp=str(torch_lib / "libomp.dylib"),
                lightgbm_rpath=lgb_rpath,
                sklearn_symlink=sk_link,
                applied_unix=time.time(),
            ),
            torch_lib,
        )
        logger.log(
            30,
            "\tmacOS OpenMP: aligned lightgbm/sklearn to torch's libomp "
            f"({torch_lib / 'libomp.dylib'}) to avoid dual-runtime segfaults "
            f"(AutoGluon#5793). Disable with {ENV_DISABLE_AUTOFIX}=1.",
        )
        _ensure_state = "fixed"
        return _ensure_state
    except Exception as e:
        logger.log(
            30,
            "\tWARNING: macOS OpenMP auto-alignment failed "
            f"({type(e).__name__}: {e}). "
            "If torch+lightgbm segfaults, ensure the env is writable and run "
            "`python -m autogluon.common.utils.macos_openmp fix`, "
            f"or set {ENV_DISABLE_AUTOFIX}=1 to silence this message.",
        )
        _ensure_state = "failed"
        return _ensure_state


def smoke() -> int:
    """Multi-thread torch matmul + lightgbm.train. Returns process-style exit code."""
    if not is_macos():
        logger.log(20, "Not macOS; OpenMP smoke skipped.")
        return 0

    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ.pop("DYLD_LIBRARY_PATH", None)
    ensure_fixed()

    try:
        import lightgbm as lgb
        import numpy as np
        import torch
    except ImportError as e:
        logger.log(40, f"smoke requires torch, lightgbm, and numpy: {e}")
        return 2

    try:
        vmmap = _run(["vmmap", str(os.getpid())])
        if vmmap.returncode == 0:
            paths = sorted(
                {line.split()[-1] for line in vmmap.stdout.splitlines() if "libomp" in line.lower() and line.strip()}
            )
            logger.log(20, f"Mapped libomp paths ({len(paths)}): {paths}")
    except OSError:
        pass

    rng = np.random.default_rng(0)
    X, y = rng.random((5000, 30)), rng.random(5000)
    logger.log(20, "Running multi-thread torch matmul...")
    (torch.rand(2000, 2000) @ torch.rand(2000, 2000)).sum().item()
    logger.log(20, "Running multi-thread lightgbm.train...")
    lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=200)
    logger.log(
        20,
        f"OpenMP smoke OK (torch={torch.__version__}, lightgbm={lgb.__version__}, platform={platform.platform()})",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry used by ``python -m autogluon.common.utils.macos_openmp``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m autogluon.common.utils.macos_openmp",
        description=(
            "Align lightgbm/sklearn OpenMP resolution to torch's libomp on macOS "
            "(relative rpath + symlink; fixes AutoGluon#5793)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p_fix = sub.add_parser("fix", help="Apply rpath/symlink alignment")
    p_fix.add_argument("--dry-run", action="store_true")
    sub.add_parser("check", help="Verify alignment")
    sub.add_parser("smoke", help="Multi-thread torch + lightgbm smoke test")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    if args.command == "fix":
        return fix(dry_run=args.dry_run)
    if args.command == "check":
        return check()
    if args.command == "smoke":
        return smoke()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
