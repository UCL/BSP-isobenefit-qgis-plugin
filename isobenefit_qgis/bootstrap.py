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
import subprocess
import sys

from qgis.PyQt.QtWidgets import QMessageBox, QWidget

CORE_IMPORT = "isobenefit"
CORE_PACKAGE = "isobenefit"
MIN_VERSION = (0, 1, 0)
MAX_VERSION_EXCLUSIVE = (0, 2, 0)
PIP_SPEC = "isobenefit>=0.1,<0.2"


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

    ``sys.executable`` is often the QGIS application binary rather than Python, so
    fall back to interpreters under the install prefixes (covers Windows OSGeo4W,
    macOS and Linux layouts).
    """
    exe = sys.executable or ""
    if os.path.basename(exe).lower().startswith("python"):
        return exe
    candidates: list[str] = []
    if os.name == "nt":
        candidates += [
            os.path.join(sys.exec_prefix, "python.exe"),
            os.path.join(sys.prefix, "python.exe"),
        ]
    else:
        for prefix in (sys.prefix, sys.exec_prefix):
            candidates += [
                os.path.join(prefix, "bin", "python3"),
                os.path.join(prefix, "bin", "python"),
            ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return exe or "python3"


def _pip_install(spec: str) -> tuple[bool, str]:
    python = _python_executable()
    attempts = [
        [python, "-m", "pip", "install", "--upgrade", spec],
        [python, "-m", "pip", "install", "--user", "--upgrade", spec],
    ]
    last_output = ""
    for cmd in attempts:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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
    prompt the user to install/upgrade and return False (a QGIS restart is needed
    before a freshly installed extension can be imported).
    """
    ok, version = core_status()
    if ok:
        return True

    if version is None:
        question = (
            f"The Isobenefit plugin needs the '{CORE_PACKAGE}' simulation engine, "
            "which is not installed.\n\nInstall it now into the QGIS Python environment?"
        )
    else:
        question = (
            f"The installed '{CORE_PACKAGE}' ({version}) is not compatible with this "
            "version of the plugin.\n\nUpgrade it now?"
        )
    choice = QMessageBox.question(
        parent,
        "Isobenefit: engine required",
        question + "\n\n(Requires an internet connection; QGIS must be restarted afterwards.)",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if choice != QMessageBox.StandardButton.Yes:
        return False

    success, output = _pip_install(PIP_SPEC)
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
    return False
