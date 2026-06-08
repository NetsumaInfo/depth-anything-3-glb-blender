@echo off
REM ============================================================
REM  Installation Depth Anything 3 -> GLB (Blender)
REM  Pre-requis: Python 3.10 + git + GPU NVIDIA (driver recent)
REM  Lance ce fichier UNE fois. Ensuite: run.bat
REM ============================================================
setlocal
title Installation Depth Anything 3
cd /d "%~dp0"

echo.
echo === [1/7] Verification Python 3.10 ===
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python 3.10 introuvable.
    echo Installe-le: https://www.python.org/downloads/release/python-31011/
    echo Coche "Add python.exe to PATH" pendant l'installation.
    pause
    exit /b 1
)

echo === [2/7] Verification git ===
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] git introuvable. Installe: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo === [3/7] Creation environnement (.venv) ===
if not exist ".venv\Scripts\python.exe" (
    py -3.10 -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip wheel setuptools

echo === [4/7] Installation PyTorch CUDA 12.1 (~2.5 Go, patiente) ===
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 ( echo [ERREUR] echec install torch & pause & exit /b 1 )

echo === [5/7] Telechargement du code Depth Anything 3 ===
if not exist "repo\pyproject.toml" (
    git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git repo
)
REM Retire xformers des dependances (pas de wheel Windows, inutile pour le modele LARGE)
powershell -NoProfile -Command "(Get-Content 'repo\pyproject.toml') | Where-Object { $_ -notmatch '\"xformers\"' } | Set-Content -Encoding utf8 'repo\pyproject.toml'"

echo === [6/7] Installation des dependances (open3d, gradio, ...) ===
set SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
pushd repo
python -m pip install -e ".[app]"
popd
if errorlevel 1 ( echo [ERREUR] echec install dependances & pause & exit /b 1 )

echo === [7/7] Dependance manquante (addict) ===
python -m pip install addict

echo.
echo ============================================================
echo  Installation terminee.
echo  Lance l'application: double-clic sur run.bat
echo  (1er lancement: telechargement du modele ~1.4 Go)
echo ============================================================
pause
