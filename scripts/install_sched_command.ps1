param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot ".."),
    [string]$UserBin = "",
    [string]$TempRoot = (Join-Path $HOME ".sched-install-temp"),
    [switch]$EditableInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PythonInvocation {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{
            InstallExecutable = $python.Source
            InstallArgs = @()
            ShimCommand = '"' + $python.Source + '"'
            Display = $python.Source
        }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @{
            InstallExecutable = $py.Source
            InstallArgs = @("-3")
            ShimCommand = '"' + $py.Source + '" -3'
            Display = $py.Source + " -3"
        }
    }

    throw "Neither 'python' nor 'py' is available on PATH."
}

function Resolve-UserBin([hashtable]$PythonInvocation, [string]$ExplicitUserBin) {
    if (-not [string]::IsNullOrWhiteSpace($ExplicitUserBin)) {
        New-Item -ItemType Directory -Force -Path $ExplicitUserBin | Out-Null
        return (Resolve-Path $ExplicitUserBin).Path
    }

    $pythonExecutable = $PythonInvocation.InstallExecutable
    $pythonRoot = Split-Path -Parent $pythonExecutable
    $scriptsDir = Join-Path $pythonRoot "Scripts"
    if (Test-Path $scriptsDir) {
        return (Resolve-Path $scriptsDir).Path
    }

    $fallback = Join-Path $HOME "bin"
    New-Item -ItemType Directory -Force -Path $fallback | Out-Null
    return (Resolve-Path $fallback).Path
}

$resolvedRepo = (Resolve-Path $RepoRoot).Path
$pythonInvocation = Resolve-PythonInvocation
$resolvedUserBin = Resolve-UserBin $pythonInvocation $UserBin

if ($EditableInstall) {
    $resolvedTempRoot = (New-Item -ItemType Directory -Force -Path $TempRoot).FullName
    $pipBuildTracker = (New-Item -ItemType Directory -Force -Path (Join-Path $resolvedTempRoot "pip-build-tracker")).FullName

    $env:TMP = $resolvedTempRoot
    $env:TEMP = $resolvedTempRoot
    $env:PIP_BUILD_TRACKER = $pipBuildTracker

    Write-Host "Installing editable SCHED command from $resolvedRepo"
    Write-Host "Using writable temp root $resolvedTempRoot"
    & $pythonInvocation.InstallExecutable @($pythonInvocation.InstallArgs) -m pip install --no-build-isolation -e $resolvedRepo
    if ($LASTEXITCODE -ne 0) {
        throw "Editable install failed with exit code $LASTEXITCODE."
    }
} else {
    Write-Host "Installing SCHED wrapper for repo-root execution from $resolvedRepo"
}

$shimPath = Join-Path $resolvedUserBin "SCHED.cmd"
$shimContent = @"
@echo off
set "SCHED_CLI_PROG=SCHED"
pushd "$resolvedRepo" >nul || exit /b 1
$($pythonInvocation.ShimCommand) -m interfaces.cli %*
set "_SCHED_EXIT=%ERRORLEVEL%"
popd >nul
exit /b %_SCHED_EXIT%
"@
Set-Content -Path $shimPath -Value $shimContent -Encoding ascii

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$pathEntries = @($userPath -split ";" | Where-Object { $_ })
$pathAlreadyPresent = $pathEntries | Where-Object {
    $_.TrimEnd("\").ToLowerInvariant() -eq $resolvedUserBin.TrimEnd("\").ToLowerInvariant()
}

if (-not $pathAlreadyPresent) {
    $newUserPath = if ([string]::IsNullOrWhiteSpace($userPath)) {
        $resolvedUserBin
    } else {
        $userPath.TrimEnd(";") + ";" + $resolvedUserBin
    }
    [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
    $env:Path = $resolvedUserBin + ";" + $env:Path
    $pathStatus = "added"
} else {
    if (-not (($env:Path -split ";") | Where-Object {
        $_.TrimEnd("\").ToLowerInvariant() -eq $resolvedUserBin.TrimEnd("\").ToLowerInvariant()
    })) {
        $env:Path = $resolvedUserBin + ";" + $env:Path
    }
    $pathStatus = "already_present"
}

Write-Host ""
Write-Host "SCHED command installed."
Write-Host "  Repo root : $resolvedRepo"
Write-Host "  Python    : $($pythonInvocation.Display)"
Write-Host "  Shim path : $shimPath"
Write-Host "  Mode      : $(if ($EditableInstall) { 'editable_install + wrapper' } else { 'wrapper_only' })"
Write-Host "  PATH      : $pathStatus"
Write-Host ""
Write-Host "If this is a new terminal command for your account, open a new shell before running 'SCHED --help'."
