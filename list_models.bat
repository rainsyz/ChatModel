@echo off
chcp 65001 > nul
@echo 正在列出可用的模型文件...
python list_models.py
@echo.
pause