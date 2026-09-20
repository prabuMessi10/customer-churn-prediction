@echo off
REM Retrain all models and recompute decision thresholds
cd /d "%~dp0"
python -m src.train_models
python -m src.optimise_thresholds
pause