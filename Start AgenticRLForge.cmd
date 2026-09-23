@echo off
setlocal
chcp 65001 >nul
title AgenticRLForge
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-studio.ps1" %*
if errorlevel 1 (
  echo.
  echo AgenticRLForge 启动失败，请查看上面的提示。
  pause
)
endlocal
