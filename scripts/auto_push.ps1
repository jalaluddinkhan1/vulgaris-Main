param(
    [string]$Branch = "main",
    [int]$IntervalSeconds = 30,
    [string]$Remote = "origin"
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path ".git")) {
    Invoke-Git init
}

Invoke-Git checkout -B $Branch

Write-Host "Watching for changes. Press Ctrl+C to stop."
Write-Host "Every detected change is committed and pushed to $Remote/$Branch."

while ($true) {
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
    }

    Start-Sleep -Seconds $IntervalSeconds
}
