"""Unit tests for autogluon.common.macos_openmp."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from autogluon.common.utils import macos_openmp as core


@pytest.fixture(autouse=True)
def _reset_state():
    core._ensure_state = None
    yield
    core._ensure_state = None


def test_is_macos():
    with mock.patch.object(core.sys, "platform", "darwin"):
        assert core.is_macos() is True
    with mock.patch.object(core.sys, "platform", "linux"):
        assert core.is_macos() is False


def test_fix_and_check_noop_off_macos():
    with mock.patch.object(core, "is_macos", return_value=False):
        assert core.fix() == 0
        assert core.check() == 0
        assert core.ensure_fixed() == "skipped"


def test_loader_relative_rpath():
    binary = Path("/env/lib/python/site-packages/lightgbm/lib/lib_lightgbm.dylib")
    torch_lib = Path("/env/lib/python/site-packages/torch/lib")
    assert core._loader_relative_rpath(binary, torch_lib) == "@loader_path/../../torch/lib"


def test_otool_libomp_deps_parse():
    out = """\
/path/lib_lightgbm.dylib:
\t@rpath/lib_lightgbm.dylib (compatibility version 0.0.0, current version 0.0.0)
\t@rpath/libomp.dylib (compatibility version 5.0.0, current version 5.0.0)
"""
    with mock.patch.object(core, "_require", return_value="otool"):
        with mock.patch.object(core, "_run", return_value=mock.Mock(returncode=0, stdout=out, stderr="")):
            assert core._otool_libomp_deps(Path("/path/x")) == ["@rpath/libomp.dylib"]


def test_otool_rpaths_parse():
    out = """\
          cmd LC_RPATH
         path /opt/homebrew/opt/libomp/lib (offset 12)
          cmd LC_RPATH
         path @loader_path/../../torch/lib (offset 12)
"""
    with mock.patch.object(core, "_require", return_value="otool"):
        with mock.patch.object(core, "_run", return_value=mock.Mock(returncode=0, stdout=out, stderr="")):
            rpaths = core._otool_rpaths(Path("/path/x"))
    assert "/opt/homebrew/opt/libomp/lib" in rpaths
    assert "@loader_path/../../torch/lib" in rpaths


def test_lightgbm_is_aligned_true_when_desired_rpath_present(tmp_path):
    torch_lib = tmp_path / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "libomp.dylib").write_bytes(b"omp")
    lgb = tmp_path / "lightgbm" / "lib" / "lib_lightgbm.dylib"
    lgb.parent.mkdir(parents=True)
    lgb.write_bytes(b"lgb")
    desired = core._loader_relative_rpath(lgb, torch_lib)

    with mock.patch.object(core, "_find_lightgbm_dylib", return_value=lgb):
        with mock.patch.object(core, "_otool_rpaths", return_value=[desired]):
            with mock.patch.object(core, "_otool_libomp_deps", return_value=["@rpath/libomp.dylib"]):
                assert core.lightgbm_is_aligned(torch_lib) is True


def test_lightgbm_is_aligned_false_with_brew_rpath(tmp_path):
    torch_lib = tmp_path / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "libomp.dylib").write_bytes(b"omp")
    lgb = tmp_path / "lightgbm" / "lib" / "lib_lightgbm.dylib"
    lgb.parent.mkdir(parents=True)
    lgb.write_bytes(b"lgb")

    with mock.patch.object(core, "_find_lightgbm_dylib", return_value=lgb):
        with mock.patch.object(core, "_otool_rpaths", return_value=["/opt/homebrew/opt/libomp/lib"]):
            with mock.patch.object(core, "_otool_libomp_deps", return_value=["@rpath/libomp.dylib"]):
                assert core.lightgbm_is_aligned(torch_lib) is False


def test_sklearn_is_aligned_symlink(tmp_path):
    torch_lib = tmp_path / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    omp = torch_lib / "libomp.dylib"
    omp.write_bytes(b"omp")
    sk = tmp_path / "sklearn" / ".dylibs" / "libomp.dylib"
    sk.parent.mkdir(parents=True)
    sk.symlink_to(omp)

    with mock.patch.object(core, "_sklearn_vendored_libomp", return_value=sk):
        assert core.sklearn_is_aligned(torch_lib) is True


def test_ensure_fixed_disabled(monkeypatch):
    monkeypatch.setenv(core.ENV_DISABLE_AUTOFIX, "1")
    with mock.patch.object(core, "is_macos", return_value=True):
        assert core.ensure_fixed() == "skipped"


def test_ensure_fixed_ok():
    with mock.patch.object(core, "is_macos", return_value=True):
        with mock.patch.object(core, "get_torch_lib_dir", return_value=Path("/tmp/torch/lib")):
            with mock.patch.object(core, "lightgbm_is_aligned", return_value=True):
                with mock.patch.object(core, "sklearn_is_aligned", return_value=True):
                    assert core.ensure_fixed() == "ok"


def test_ensure_fixed_applies(tmp_path):
    torch_lib = tmp_path / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "libomp.dylib").write_bytes(b"omp")
    with mock.patch.object(core, "is_macos", return_value=True):
        with mock.patch.object(core, "get_torch_lib_dir", return_value=torch_lib):
            with mock.patch.object(core, "lightgbm_is_aligned", return_value=False):
                with mock.patch.object(core, "sklearn_is_aligned", return_value=True):
                    with mock.patch.object(core, "_align_lightgbm", return_value="@loader_path/../../torch/lib"):
                        with mock.patch.object(core, "_align_sklearn", return_value=str(torch_lib / "libomp.dylib")):
                            with mock.patch.object(core, "_find_lightgbm_dylib", return_value=None):
                                assert core.ensure_fixed() == "fixed"


def test_ensure_fixed_failed():
    with mock.patch.object(core, "is_macos", return_value=True):
        with mock.patch.object(core, "get_torch_lib_dir", return_value=Path("/tmp/torch/lib")):
            with mock.patch.object(core, "lightgbm_is_aligned", return_value=False):
                with mock.patch.object(core, "sklearn_is_aligned", return_value=False):
                    with mock.patch.object(core, "_align_lightgbm", side_effect=RuntimeError("perm")):
                        assert core.ensure_fixed() == "failed"


def test_main_fix_dry_run():
    from autogluon.common.utils.macos_openmp import main

    with mock.patch("autogluon.common.utils.macos_openmp.fix", return_value=0) as fix:
        assert main(["fix", "--dry-run"]) == 0
        fix.assert_called_once_with(dry_run=True)


@pytest.mark.skipif(not core.is_macos(), reason="macOS only")
def test_integration_check_and_smoke():
    from autogluon.common.utils.macos_openmp import main

    assert main(["fix"]) in (0, 1, 2)
    assert main(["check"]) in (0, 1, 2)
    # smoke is heavier; only require it does not ImportError in harness
    assert main(["smoke"]) in (0, 2)
