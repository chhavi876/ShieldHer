# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata, collect_data_files
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Collect all necessary data files and metadata
datas = []
datas += copy_metadata('streamlit')
datas += copy_metadata('plotly')
datas += copy_metadata('pandas')
datas += copy_metadata('numpy')
datas += copy_metadata('PIL')
datas += copy_metadata('cv2')
datas += copy_metadata('speech_recognition')

# Include Streamlit static files
datas += collect_data_files('streamlit')
datas += collect_data_files('plotly')

# Include additional necessary files
datas += [('data', 'data')]  # Include data directory if it exists

# Hidden imports - all modules that PyInstaller might miss
hiddenimports = [
    'streamlit',
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner.script_runner',
    'streamlit.runtime.state.session_state',
    'plotly',
    'plotly.graph_objects',
    'plotly.express',
    'plotly.subplots',
    'pandas',
    'numpy',
    'PIL',
    'PIL.Image',
    'cv2',
    'speech_recognition',
    'requests',
    'json',
    'threading',
    'time',
    'datetime',
    'os',
    'base64',
    'io',
    'altair',
    'tornado',
    'tornado.web',
    'tornado.websocket',
    'tornado.httpserver',
    'tornado.ioloop',
    'click',
    'blinker',
    'cachetools',
    'importlib_metadata',
    'packaging',
    'pyarrow',
    'tzlocal',
    'validators',
    'watchdog',
    'gitpython',
    'pympler',
    'rich',
    'toml',
    'typing_extensions',
    'urllib3',
    'certifi',
    'charset_normalizer',
    'idna'
]

# Analysis configuration
a = Analysis(
    ['shieldher_app.py'],  # Your main Streamlit app file
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate entries
a.datas = list(set(a.datas))

# PYZ configuration
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# EXE configuration
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ShieldHer',
    debug=False,
    bootloader_ignore_signals=False,
    strip
)