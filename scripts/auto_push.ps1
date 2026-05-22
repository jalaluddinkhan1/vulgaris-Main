param(
    [string]$Branch = "main",
    [int]$IntervalSeconds = 30,
    [string]$Remote = "origin",
    [switch]$Once
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path ".git")) {
    Invoke-Git init
}

Invoke-Git checkout -B $Branch

Write-Host "Repository: $RepoRoot"
Write-Host "Branch: $Branch"
Write-Host "Remote: $Remote"

function Sync-Changes {
    $changes = git status --porcelain

    if ($changes) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
        Invoke-Git add -A

        $staged = git diff --cached --name-only
        if ($staged) {
            Invoke-Git commit -m "Auto update: $timestamp"
            Invoke-Git push -u $Remote $Branch
            Write-Host "Pushed auto update at $timestamp"
        }
    } else {
        Write-Host "No changes to push."
    }
}

if ($Once) {
    Sync-Changes
    exit 0
}

Write-Host "Watching for changes. Press Ctrl+C to stop."
Write-Host "Every detected change is committed and pushed to $Remote/$Branch."

while ($true) {
    Sync-Changes
    Start-Sleep -Seconds $IntervalSeconds
}
