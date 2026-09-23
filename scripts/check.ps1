[CmdletBinding()]
param(
    [switch]$Security,
    [switch]$SkipPackage
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )

    Write-Step $Description
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE. Review the command output above."
    }
}

function Test-SupportedVenv {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string]$VenvDirectory
    )

    $probe = @"
import os
import sys
expected = os.path.normcase(os.path.realpath(sys.argv[1]))
actual = os.path.normcase(os.path.realpath(sys.prefix))
supported = (3, 10) <= sys.version_info[:2] <= (3, 12)
raise SystemExit(0 if supported and sys.prefix != sys.base_prefix and actual == expected else 1)
"@
    try {
        & $FilePath "-c" $probe $VenvDirectory 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Find-VenvPython {
    param([Parameter(Mandatory)][string]$VenvDirectory)

    foreach ($relativePath in @("Scripts\python.exe", "bin/python")) {
        $candidate = Join-Path $VenvDirectory $relativePath
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    return $null
}

function Find-GitBash {
    $candidates = @()
    if ($env:ProgramFiles) {
        $candidates += Join-Path $env:ProgramFiles "Git\bin\bash.exe"
    }
    if (${env:ProgramFiles(x86)}) {
        $candidates += Join-Path ${env:ProgramFiles(x86)} "Git\bin\bash.exe"
    }
    if ($env:LOCALAPPDATA) {
        $candidates += Join-Path $env:LOCALAPPDATA "Programs\Git\bin\bash.exe"
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    return $null
}

$distDirectory = $null
try {
    $projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
    $pyproject = Join-Path $projectRoot "pyproject.toml"
    $venvDirectory = Join-Path $projectRoot ".venv"

    if (-not (Test-Path -LiteralPath $pyproject -PathType Leaf)) {
        throw "pyproject.toml was not found at $projectRoot. Use a complete project checkout."
    }
    if (Test-Path -LiteralPath $venvDirectory) {
        $venvItem = Get-Item -LiteralPath $venvDirectory -Force
        if (($venvItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$venvDirectory is a reparse point; refusing to use a linked environment."
        }
    }

    $python = Find-VenvPython -VenvDirectory $venvDirectory
    if ($null -eq $python) {
        throw "No project environment was found. Run $(Join-Path $PSScriptRoot 'setup.ps1') first."
    }
    if (-not (Test-SupportedVenv -FilePath $python -VenvDirectory $venvDirectory)) {
        throw ".venv is not a valid project-local Python 3.10-3.12 environment. Run setup.ps1 -Recreate."
    }

    $moduleProbe = @"
import importlib.util
import sys
required = ("ruff", "mypy", "pytest", "build", "twine")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing development modules: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
"@
    & $python "-c" $moduleProbe
    if ($LASTEXITCODE -ne 0) {
        throw "Development dependencies are incomplete. Rerun setup.ps1 without -RuntimeOnly."
    }

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    Push-Location -LiteralPath $projectRoot
    try {
        Invoke-Native -Description "Checking dependency consistency" -FilePath $python `
            -Arguments @("-m", "pip", "check")
        Invoke-Native -Description "Linting Python sources" -FilePath $python `
            -Arguments @("-m", "ruff", "check", ".")
        Invoke-Native -Description "Checking Python formatting" -FilePath $python `
            -Arguments @("-m", "ruff", "format", "--check", "src", "tests", "examples")
        Invoke-Native -Description "Running strict type checks" -FilePath $python `
            -Arguments @("-m", "mypy", "src", "examples/offline_pipeline.py")
        Invoke-Native -Description "Running tests with coverage" -FilePath $python `
            -Arguments @("-m", "pytest", "--cov=agentic_rl_forge", "--cov-report=term-missing")

        $gitBash = Find-GitBash
        if ($null -ne $gitBash) {
            Invoke-Native -Description "Checking the verl shell recipe syntax" -FilePath $gitBash `
                -Arguments @("-n", "recipes/verl/run_search_r1_grpo.sh")
        }
        else {
            Write-Warning "Git Bash was not found; the Linux-only verl recipe syntax check was skipped. CI will still validate it."
        }

        if (-not $SkipPackage) {
            $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
            $distDirectory = Join-Path $tempRoot ("arf-quality-dist-" + [guid]::NewGuid().ToString("N"))
            $null = New-Item -ItemType Directory -Path $distDirectory

            Invoke-Native -Description "Building wheel and source distribution" -FilePath $python `
                -Arguments @("-m", "build", "--outdir", $distDirectory)
            $packages = @(Get-ChildItem -LiteralPath $distDirectory -File | ForEach-Object FullName)
            if ($packages.Count -eq 0) {
                throw "Package build produced no files in $distDirectory."
            }
            $twineArguments = @("-m", "twine", "check", "--strict") + $packages
            Invoke-Native -Description "Validating package metadata" -FilePath $python `
                -Arguments $twineArguments
        }

        if ($Security) {
            & $python "-c" "import pip_audit" 2>$null
            if ($LASTEXITCODE -ne 0) {
                throw "pip-audit is not installed. Rerun setup.ps1 without -RuntimeOnly."
            }
            Invoke-Native -Description "Auditing installed dependencies" -FilePath $python `
                -Arguments @("-m", "pip_audit")
        }
    }
    finally {
        Pop-Location
    }

    $stopwatch.Stop()
    Write-Host ""
    Write-Host ("All requested quality checks passed in {0:N1} seconds." -f $stopwatch.Elapsed.TotalSeconds) `
        -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    if ($null -ne $distDirectory -and (Test-Path -LiteralPath $distDirectory -PathType Container)) {
        $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
        $resolvedDist = [System.IO.Path]::GetFullPath($distDirectory)
        $expectedPrefix = Join-Path $tempRoot "arf-quality-dist-"
        if ($resolvedDist.StartsWith($expectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolvedDist -Recurse -Force
        }
    }
}
