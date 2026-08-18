[CmdletBinding()]
param(
    [string]$StateFile = "state/remote-execution.json",
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-NativeText {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    $output = & $FilePath @ArgumentList 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE`: $($output -join [Environment]::NewLine)"
    }
    return ($output -join "`n").Trim()
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$gitRoot = Get-NativeText -FilePath "git" -ArgumentList @("-C", $repoRoot, "rev-parse", "--show-toplevel")
if ([IO.Path]::GetFullPath($gitRoot) -ne $repoRoot) {
    throw "Script must run from the xh-202628-agent Git repository."
}

$statePath = if ([IO.Path]::IsPathRooted($StateFile)) {
    $StateFile
} else {
    Join-Path $repoRoot $StateFile
}
$statePath = (Resolve-Path -LiteralPath $statePath).Path
$config = Get-Content -Raw -Encoding utf8 -LiteralPath $statePath | ConvertFrom-Json

if ($config.schema_version -ne 1) {
    throw "Unsupported remote state schema: $($config.schema_version)"
}
if (-not $config.directory_creation.authorized) {
    throw "Remote directory creation is not authorized in state/remote-execution.json."
}
if ($config.directory_creation.created) {
    throw "State already records the remote mirror as created. Refusing to bootstrap again."
}
if ([string]::IsNullOrWhiteSpace([string]$config.directory_creation.authorization_quote) -or
    [string]::IsNullOrWhiteSpace([string]$config.directory_creation.authorized_at)) {
    throw "Authorization quote and timestamp must be recorded before remote creation."
}

$sshTarget = [string]$config.ssh_target
$remoteBase = [string]$config.base_directory
if ($sshTarget -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$') {
    throw "Unsafe SSH target in state file: $sshTarget"
}
if ($remoteBase -notmatch '^/[A-Za-z0-9._/-]+$' -or $remoteBase -match '/\.\.?(/|$)') {
    throw "Unsafe remote base directory in state file: $remoteBase"
}
if ($remoteBase -ne "/home/user/lynsdu2/xh-202628-agent") {
    throw "Remote base differs from the approved project path: $remoteBase"
}

$status = Get-NativeText -FilePath "git" -ArgumentList @("-C", $repoRoot, "status", "--porcelain=v1", "--untracked-files=all")
if (-not [string]::IsNullOrWhiteSpace($status)) {
    throw "Commit the authorization record and start from a clean worktree before remote creation.`n$status"
}
$commit = Get-NativeText -FilePath "git" -ArgumentList @("-C", $repoRoot, "rev-parse", "HEAD")

Write-Host "Approved target: $sshTarget`:$remoteBase"
Write-Host "Authorization is recorded in commit: $commit"
if (-not $Apply) {
    Write-Host "Dry run only. Re-run with -Apply after checking the exact target."
    exit 0
}

$preflightCommand = "test ! -e '$remoteBase'"
& ssh $sshTarget $preflightCommand
if ($LASTEXITCODE -ne 0) {
    throw "Remote target already exists or could not be checked. Nothing was created."
}

$marker = "xh-202628-execution-mirror-v1"
$createCommand = "set -eu; umask 077; base='$remoteBase'; test ! -e `"`$base`"; mkdir -m 700 -- `"`$base`"; mkdir -m 700 -- `"`$base/incoming`" `"`$base/locks`" `"`$base/runs`"; printf '%s\n' '$marker' > `"`$base/.xh-202628-execution-mirror`"; stat -c '%n %U:%G %a' -- `"`$base`" `"`$base/incoming`" `"`$base/locks`" `"`$base/runs`""
& ssh $sshTarget $createCommand
if ($LASTEXITCODE -ne 0) {
    throw "Remote bootstrap failed. Preserve the partial state and report it; do not auto-clean."
}

Write-Host "Remote execution mirror created. Record created=true/created_at in state files and commit that update."
