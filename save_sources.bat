@echo off
setlocal enabledelayedexpansion

REM Racine = dossier du script (chemin absolu normalisé)
set "ROOT_DIR=%~dp0" & for %%A in ("%ROOT_DIR:~0,-1%") do set "ROOT_DIR=%%~fA"
set "INCLUDE_DIR=src"

REM Timestamp (approx: YYYYMMDD_HHMM)
for /f "tokens=1-4 delims=/- " %%a in ("%date%") do set "DATE=%%d%%b%%c"
for /f "tokens=1-2 delims=:." %%h in ("%time%") do set "TIME=%%h%%i"
set "DATE=%DATE: =0%" & set "TIME=%TIME: =0%"
set "TS=%DATE%_%TIME%"

set "ARCHIVE_NAME=code_sources_%TS%.zip"
set "ARCHIVE_PATH=%ROOT_DIR%\%ARCHIVE_NAME%"

REM Fichier temporaire listant les fichiers à archiver (chemins relatifs)
set "TMP_LIST=%TEMP%\files_to_zip_%RANDOM%.lst"
if exist "%TMP_LIST%" del /q "%TMP_LIST%" >nul 2>&1

pushd "%ROOT_DIR%"

REM 1) Ajouter les fichiers .py / .yaml / .yml à la racine (non récursif)
for %%F in (*.py *.yaml *.yml) do (
  if exist "%%F" echo .\%%F>> "%TMP_LIST%"
)

REM 2) Ajouter les fichiers sous src (récursif) si src existe
if exist "%INCLUDE_DIR%" (
  for /f "usebackq delims=" %%F in (`dir /b /s "%INCLUDE_DIR%\*.py" "%INCLUDE_DIR%\*.yaml" "%INCLUDE_DIR%\*.yml" 2^>nul`) do (
    REM Convertir absolu -> relatif à ROOT_DIR
    set "ABS=%%~fF"
    set "REL=!ABS:%ROOT_DIR%=.!"
    echo !REL!>> "%TMP_LIST%"
  )
)

REM Vérifier PowerShell et Compress-Archive
where powershell >nul 2>&1
if errorlevel 1 goto TRY_7Z

REM Construire une commande PowerShell pour zipper la liste
REM Compress-Archive requiert -Path avec un tableau; on lit la liste et la passe à Compress-Archive
powershell -NoLogo -NoProfile -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$root = Get-Location;" ^
  "$paths = Get-Content -LiteralPath '%TMP_LIST%';" ^
  "if (Test-Path -LiteralPath '%ARCHIVE_PATH%') { Remove-Item -LiteralPath '%ARCHIVE_PATH%' -Force; }" ^
  "Compress-Archive -Path $paths -DestinationPath '%ARCHIVE_PATH%' -CompressionLevel Optimal;"
if errorlevel 1 (
  echo Erreur: creation ZIP via PowerShell echouee. Tentative avec 7-Zip...
  goto TRY_7Z
) else (
  del /q "%TMP_LIST%" >nul 2>&1
  echo Archive creee: %ARCHIVE_PATH%
  popd
  exit /b 0
)

:TRY_7Z
where 7z >nul 2>&1
if errorlevel 1 (
  echo Aucun outil ZIP disponible (PowerShell Compress-Archive ni 7z).
  del /q "%TMP_LIST%" >nul 2>&1
  popd
  exit /b 1
) else (
  REM 7z accepte une liste avec l'option @
  if exist "%ARCHIVE_PATH%" del /q "%ARCHIVE_PATH%" >nul 2>&1
  7z a -tzip "%ARCHIVE_PATH%" @"%TMP_LIST%" >nul
  if errorlevel 1 (
    echo Erreur: creation ZIP via 7-Zip echouee.
    del /q "%TMP_LIST%" >nul 2>&1
    popd
    exit /b 1
  )
  del /q "%TMP_LIST%" >nul 2>&1
  echo Archive creee: %ARCHIVE_PATH%
  popd
  exit /b 0
)
