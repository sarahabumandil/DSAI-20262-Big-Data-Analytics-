@echo off
setlocal

set "WORKINGDIRECTORY="C:\WPy64-3.13.12.0\notebooks\Dev\bda"
::set "VENV=C:\WPy64-3.13.12.0\notebooks\Envs\bigdata"
set "VENV=C:\WPy64-3.13.12.0\Envs\bigdata"

cd /d "%WORKINGDIRECTORY%"
call "%VENV%\Scripts\activate.bat"

rem optional sanity checks
python -V
where python

rem start Jupyter in this venv
jupyter notebook
