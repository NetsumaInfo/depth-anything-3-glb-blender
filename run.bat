@echo off
REM ============================================================
REM  Depth Anything 3  ->  GLB pour Blender  (interface Gradio)
REM  Double-clique ce fichier pour lancer l'app.
REM ============================================================
title Depth Anything 3 - GLB Blender
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERREUR] venv introuvable: .venv\Scripts\python.exe
    echo Installation incomplete. Relance l'installation.
    pause
    exit /b 1
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo Lancement... l'interface s'ouvre sur http://127.0.0.1:7860
echo (Premier lancement: telechargement du modele ~1.4 Go, patiente.)
echo.

".venv\Scripts\python.exe" app.py

echo.
echo App arretee.
pause
