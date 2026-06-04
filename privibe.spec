# -*- mode: python ; coding: utf-8 -*-
# Onefile build for privibe CLI.
# Build: uv run --group build pyinstaller privibe.spec

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

core_builtins_deps = collect_all('privibe.core.tools.builtins')

hidden_imports = ["truststore"] + collect_submodules("rich._unicode_data")
for item in core_builtins_deps[2]:
    if isinstance(item, str):
        hidden_imports.append(item)

binaries = core_builtins_deps[1]

# Auto-discover all sample skills under privibe/sample_skills/*/
sample_skills_datas = []
for skill_dir in sorted(Path('privibe/sample_skills').iterdir()):
    if skill_dir.is_dir():
        for f in skill_dir.rglob('*'):
            if f.is_file():
                sample_skills_datas.append((str(f), str(f.parent)))

a = Analysis(
    ['privibe/cli/entrypoint.py'],
    pathex=[],
    binaries=binaries,
    datas=[
        ('privibe/core/prompts/*.md', 'privibe/core/prompts'),
        ('privibe/core/tools/builtins/prompts/*.md', 'privibe/core/tools/builtins/prompts'),
        ('privibe/setup/*', 'privibe/setup'),
        ('privibe/setup/**/*.tcss', 'privibe/setup'),
        ('privibe/core/tools/builtins/*.py', 'privibe/core/tools/builtins'),
        ('privibe/cli/textual_ui/*.tcss', 'privibe/cli/textual_ui'),
    ] + sample_skills_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["pyinstaller/runtime_hook_truststore.py"],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='privibe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
