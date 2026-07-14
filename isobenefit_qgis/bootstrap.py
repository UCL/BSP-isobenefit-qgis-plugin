"""Runtime dependency bootstrap for the Isobenefit simulation core.

The QGIS plugin repository does not allow shipping binaries, so the compiled Rust
core (``isobenefit``) cannot live in the plugin zip. Instead, on use, the
plugin checks whether the core is importable and version-compatible and, if not,
offers to ``pip install`` it into the QGIS Python environment.

This module must stay import-safe: it never imports ``isobenefit`` at module
load (the real import happens lazily in ``sim_runner``), so the plugin always
loads in QGIS even when the core is absent.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys

from qgis.core import Qgis, QgsApplication, QgsMessageLog, QgsTask
from qgis.PyQt.QtWidgets import QMessageBox, QWidget

CORE_IMPORT = "isobenefit"
CORE_PACKAGE = "isobenefit"
# the floor tracks the oldest core this plugin should drive: 0.12.17 adds the
# engine-side walk_distance field that makes post-processing fast on large windows
# (the plugin falls back to a slow Python walk on older engines)
MIN_VERSION = (0, 12, 17)
MAX_VERSION_EXCLUSIVE = (0, 13, 0)
PIP_SPEC = "isobenefit>=0.12.17,<0.13"

# module-level so the running install survives ensure_core returning
_install_task: QgsTask | None = None


def _parse_version(version: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for chunk in version.split(".")[:3]:
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def core_status() -> tuple[bool, str | None]:
    """Return ``(ok, installed_version)``.

    ``ok`` is False if the core is missing or its version is outside the supported
    range. ``installed_version`` is None when nothing is importable.
    """
    if importlib.util.find_spec(CORE_IMPORT) is None:
        return False, None
    try:
        module = importlib.import_module(CORE_IMPORT)
        version = str(getattr(module, "__version__", "0.0.0"))
    except Exception:
        return False, None
    parsed = _parse_version(version)
    if parsed < MIN_VERSION or parsed >= MAX_VERSION_EXCLUSIVE:
        return False, version
    return True, version


def _python_executable() -> str:
    """Best-effort path to the QGIS Python interpreter to drive pip.

    On some platforms (notably macOS) ``sys.executable`` is the host *application*
    binary rather than Python, and ``sys.prefix`` may point at a build path that
    does not exist locally. We therefore prefer ``sys._base_executable`` (the real
    interpreter) and ONLY ever return a python-named binary — never the host app,
    which would otherwise try to open the pip arguments as data sources.
    """
    base = getattr(sys, "_base_executable", "") or ""
    names = (
        ["python.exe", "python3.exe"]
        if os.name == "nt"
        else [
            "python3",
            "python3.13",
            "python3.12",
            "python3.11",
            "python3.10",
            "python3.9",
            "python",
        ]
    )
    candidates: list[str] = [base, sys.executable or ""]
    for ref in (base, sys.executable or ""):
        ref_dir = os.path.dirname(ref)
        if ref_dir:
            candidates += [os.path.join(ref_dir, n) for n in names]
    for prefix in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
        if os.name == "nt":
            candidates.append(os.path.join(prefix, "python.exe"))
        else:
            candidates += [os.path.join(prefix, "bin", n) for n in ("python3", "python")]
    for candidate in candidates:
        if candidate and os.path.basename(candidate).lower().startswith("python") and os.path.exists(candidate):
            return candidate
    # never fall back to the host app binary; defer to PATH instead
    return "python3"


def _subprocess_env() -> dict:
    """Environment for the pip subprocess.

    A bundled interpreter may not boot standalone unless ``PYTHONHOME`` is set,
    because the host can configure it via the C API rather than env vars. Derive
    the home from the running stdlib location (don't override an existing value).
    """
    env = os.environ.copy()
    try:
        stdlib = os.path.dirname(os.__file__)  # <prefix>/lib/pythonX.Y
        home = os.path.dirname(os.path.dirname(stdlib))  # <prefix>
        if os.path.isdir(stdlib) and "PYTHONHOME" not in env:
            env["PYTHONHOME"] = home
    except Exception as exc:  # unusual layout; pip may still boot without PYTHONHOME
        QgsMessageLog.logMessage(
            f"Could not derive PYTHONHOME for the pip subprocess: {exc}",
            "Isobenefit",
            Qgis.MessageLevel.Warning,
        )
    return env


def _profile_python_dir() -> str | None:
    """The active QGIS profile's ``python/`` directory.

    QGIS prepends this to ``sys.path``, so a package installed here is the copy that
    actually gets imported — site-packages can be read-only, or shadowed by a copy
    here. Returns None if it can't be determined.
    """
    try:
        profile = QgsApplication.qgisSettingsDirPath()
        if profile:
            return os.path.join(profile, "python")
    except Exception as exc:  # falls back to a plain pip install below
        QgsMessageLog.logMessage(
            f"Could not determine the QGIS profile python/ directory: {exc}",
            "Isobenefit",
            Qgis.MessageLevel.Warning,
        )
    return None


def _pip_install(spec: str) -> tuple[bool, str]:
    python = _python_executable()
    env = _subprocess_env()
    target = _profile_python_dir()
    if target:
        # Install into the profile's python/ dir so the new copy is the one QGIS
        # imports. Clear any prior copy first so an upgrade truly replaces it — a stale
        # (e.g. hand-placed) copy here would otherwise shadow the new one and the
        # version check would loop forever.
        try:
            os.makedirs(target, exist_ok=True)
            for name in os.listdir(target):
                if name == CORE_PACKAGE or (name.startswith(CORE_PACKAGE + "-") and name.endswith(".dist-info")):
                    shutil.rmtree(os.path.join(target, name), ignore_errors=True)
        except Exception as exc:  # pip --target --upgrade will still overwrite in place
            QgsMessageLog.logMessage(
                f"Could not clear a previous engine copy from {target}: {exc}",
                "Isobenefit",
                Qgis.MessageLevel.Warning,
            )
        attempts = [
            [python, "-m", "pip", "install", "--target", target, "--upgrade", "--no-deps", spec],
        ]
    else:
        attempts = [
            [python, "-m", "pip", "install", "--upgrade", spec],
            [python, "-m", "pip", "install", "--user", "--upgrade", spec],
        ]
    last_output = ""
    for cmd in attempts:
        try:
            # a fixed argument list (resolved interpreter, -m pip install, constant
            # version spec), no shell, no user-supplied input
            result = subprocess.run(  # nosec B603
                cmd, capture_output=True, text=True, timeout=600, env=env
            )
        except Exception as exc:  # network, permissions, missing pip, ...
            last_output = str(exc)
            continue
        if result.returncode == 0:
            return True, result.stdout
        last_output = (result.stdout or "") + "\n" + (result.stderr or "")
    return False, last_output


def ensure_core(parent: QWidget | None = None) -> bool:
    """Ensure the core is importable and compatible.

    Returns True if it is usable right now. If it is missing or out of range,
    offer to install/upgrade it as a background task and return False (a QGIS
    restart is needed before a freshly installed extension can be imported). The
    pip subprocess runs in a QgsTask so the UI stays responsive on slow networks.
    """
    global _install_task
    ok, version = core_status()
    if ok:
        return True

    if _install_task is not None:
        QgsMessageLog.logMessage(
            "The engine install is still running; watch the task bar / message log.",
            level=Qgis.MessageLevel.Info,
            notifyUser=True,
        )
        return False

    if version is None:
        question = (
            f"The Isobenefit plugin needs the '{CORE_PACKAGE}' simulation engine, "
            "which is not installed.\n\nDownload and install it now into the QGIS "
            "Python environment?"
        )
    else:
        question = (
            f"The installed '{CORE_PACKAGE}' ({version}) is not compatible with this "
            "version of the plugin.\n\nUpgrade it now?"
        )
    choice = QMessageBox.question(
        parent,
        "Isobenefit: engine required",
        question
        + "\n\n(Requires an internet connection. The download runs in the background "
        "and you will be notified; QGIS must be restarted afterwards.)",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if choice != QMessageBox.StandardButton.Yes:
        return False

    def _install(_task: QgsTask):
        return _pip_install(PIP_SPEC)

    def _done(exception, result=None):
        global _install_task
        _install_task = None
        success, output = (False, str(exception)) if exception is not None else result
        if success:
            QMessageBox.information(
                parent,
                "Isobenefit: engine installed",
                f"'{CORE_PACKAGE}' was installed. Please restart QGIS, then run the plugin again.",
            )
        else:
            python = _python_executable()
            QMessageBox.critical(
                parent,
                "Isobenefit: install failed",
                "Automatic installation failed. Install it manually from a terminal:\n\n"
                f'    "{python}" -m pip install "{PIP_SPEC}"\n\n'
                "Then restart QGIS.\n\nDetails:\n" + (output[-1500:] if output else "(no output)"),
            )

    _install_task = QgsTask.fromFunction("Installing the Isobenefit engine", _install, on_finished=_done)
    QgsApplication.taskManager().addTask(_install_task)
    return False
