[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^exp-[0-9]{8}-[0-9]{3}$')]
    [string]$ExperimentId,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^remote-jobs/[A-Za-z0-9._/-]+[.]sh$')]
    [string]$EntryPoint,

    [Parameter(Mandatory = $true)]
    [int[]]$GpuIds,

    [ValidatePattern('^exp-[0-9]{8}-[0-9]{3}-[0-9a-f]{12}$')]
    [string]$RetrieveExistingRun,

    [string]$StateFile = "state/remote-execution.json"
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

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
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
if (-not $config.directory_creation.authorized -or -not $config.directory_creation.created) {
    throw "Remote execution mirror has not been authorized and recorded as created."
}
if (-not $config.execution.enabled) {
    throw "Remote GPU execution is disabled in state/remote-execution.json."
}
if ([string]::IsNullOrWhiteSpace([string]$config.execution.authorization_quote) -or
    [string]::IsNullOrWhiteSpace([string]$config.execution.authorized_at)) {
    throw "GPU policy authorization quote and timestamp must be recorded before execution."
}
$requiredPolicy = @(
    "max_start_gpu_utilization_percent",
    "max_start_memory_used_mib",
    "max_run_seconds",
    "max_remote_storage_gib"
)
foreach ($field in $requiredPolicy) {
    if ($null -eq $config.execution.$field) {
        throw "Remote policy field '$field' must be set before execution."
    }
}

$sshTarget = [string]$config.ssh_target
$remoteBase = [string]$config.base_directory
$cudaHome = [string]$config.toolchain.cuda_home
if ($sshTarget -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$') {
    throw "Unsafe SSH target in state file: $sshTarget"
}
if ($remoteBase -notmatch '^/[A-Za-z0-9._/-]+$' -or $remoteBase -match '/\.\.?(/|$)') {
    throw "Unsafe remote base directory in state file: $remoteBase"
}
if ($remoteBase -ne "/home/user/lynsdu2/xh-202628-agent") {
    throw "Remote base differs from the approved project path: $remoteBase"
}
if ($cudaHome -notmatch '^/[A-Za-z0-9._/-]+$') {
    throw "Unsafe CUDA home in state file: $cudaHome"
}
if ($cudaHome -ne "/usr/local/cuda") {
    throw "This server must use the verified CUDA 12.2 toolchain at /usr/local/cuda."
}

$maxUtil = [int]$config.execution.max_start_gpu_utilization_percent
$maxMemoryMiB = [int]$config.execution.max_start_memory_used_mib
$maxRunSeconds = [int]$config.execution.max_run_seconds
$maxStorageGiB = [int]$config.execution.max_remote_storage_gib
$maxParallelRuns = [int]$config.execution.max_parallel_runs
if ($maxUtil -lt 0 -or $maxUtil -gt 100) {
    throw "max_start_gpu_utilization_percent must be between 0 and 100."
}
if ($maxMemoryMiB -lt 0 -or $maxRunSeconds -le 0 -or $maxStorageGiB -le 0) {
    throw "Memory, run-time, and storage policy values must be non-negative/positive."
}
if ($maxParallelRuns -lt 1 -or $maxParallelRuns -gt 8) {
    throw "max_parallel_runs must be between 1 and 8."
}

$entryPointNormalized = $EntryPoint.Replace('\', '/')
if ($entryPointNormalized.Split('/') -contains '..' -or $entryPointNormalized.Split('/') -contains '.') {
    throw "EntryPoint cannot contain dot-directory traversal."
}

$requestedGpuIds = @($GpuIds | Sort-Object -Unique)
if ($requestedGpuIds.Count -eq 0 -or @($requestedGpuIds | Where-Object { $_ -lt 0 }).Count -gt 0) {
    throw "At least one non-negative GPU ID is required."
}
$allowedGpuIds = @($config.execution.allowed_gpu_ids | ForEach-Object { [int]$_ })
if (@($allowedGpuIds | Where-Object { $_ -lt 0 -or $_ -gt 7 }).Count -gt 0) {
    throw "This verified server exposes GPU IDs 0 through 7 only."
}
foreach ($gpuId in $requestedGpuIds) {
    if ($gpuId -notin $allowedGpuIds) {
        throw "GPU $gpuId is not in the user-approved allowed_gpu_ids list."
    }
}

$localResultParent = Join-Path $repoRoot "artifacts/raw/remote-runs"
if (-not [string]::IsNullOrWhiteSpace($RetrieveExistingRun)) {
    if (-not $RetrieveExistingRun.StartsWith("$ExperimentId-")) {
        throw "RetrieveExistingRun must belong to ExperimentId $ExperimentId."
    }
    $remoteRun = "$remoteBase/runs/$RetrieveExistingRun"
    $remoteStatusCommand =
        "grep -qx 'xh-202628-execution-mirror-v1' '$remoteBase/.xh-202628-execution-mirror' && " +
        "test -f '$remoteRun/results/status.json' && cat '$remoteRun/results/status.json'"
    $remoteStatusText = Get-NativeText -FilePath "ssh" -ArgumentList @(
        "-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR", $sshTarget, $remoteStatusCommand
    )
    $remoteStatus = $remoteStatusText | ConvertFrom-Json
    $expectedShortCommit = $RetrieveExistingRun.Substring($RetrieveExistingRun.Length - 12)
    if ([string]$remoteStatus.run_id -ne $RetrieveExistingRun) {
        throw "Remote status run_id does not match the requested run."
    }
    if (-not ([string]$remoteStatus.commit).StartsWith($expectedShortCommit)) {
        throw "Remote status commit does not match the run ID suffix."
    }
    if ([string]$remoteStatus.state -notin @("succeeded", "failed", "rejected")) {
        throw "Remote run has not reached a terminal state: $($remoteStatus.state)"
    }

    $localResultDir = Join-Path $localResultParent $RetrieveExistingRun
    if (Test-Path -LiteralPath $localResultDir) {
        throw "Local result directory already exists: $localResultDir"
    }
    New-Item -ItemType Directory -Force -Path $localResultParent | Out-Null
    $remoteResults = $sshTarget + ":" + $remoteRun + "/results"
    Invoke-NativeChecked -FilePath "scp" -ArgumentList @(
        "-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR", "-r",
        $remoteResults,
        $localResultDir
    )
    $localStatusPath = Join-Path $localResultDir "status.json"
    $localStatus = Get-Content -Raw -Encoding utf8 -LiteralPath $localStatusPath | ConvertFrom-Json
    if ([string]$localStatus.run_id -ne $RetrieveExistingRun -or
        [string]$localStatus.commit -ne [string]$remoteStatus.commit -or
        [string]$localStatus.state -ne [string]$remoteStatus.state) {
        throw "Retrieved status does not match the verified remote status."
    }
    Write-Host "Recovered remote run: $RetrieveExistingRun"
    Write-Host "Commit: $($remoteStatus.commit)"
    Write-Host "Local raw results: $localResultDir"
    return
}

$status = Get-NativeText -FilePath "git" -ArgumentList @("-C", $repoRoot, "status", "--porcelain=v1", "--untracked-files=all")
if (-not [string]::IsNullOrWhiteSpace($status)) {
    throw "Remote tests require a completely clean, committed worktree.`n$status"
}

$commit = Get-NativeText -FilePath "git" -ArgumentList @("-C", $repoRoot, "rev-parse", "HEAD")
$shortCommit = Get-NativeText -FilePath "git" -ArgumentList @("-C", $repoRoot, "rev-parse", "--short=12", "HEAD")
Invoke-NativeChecked -FilePath "git" -ArgumentList @("-C", $repoRoot, "cat-file", "-e", "HEAD:$entryPointNormalized")
Invoke-NativeChecked -FilePath "git" -ArgumentList @("-C", $repoRoot, "cat-file", "-e", "HEAD:scripts/remote-stage.sh")
Invoke-NativeChecked -FilePath "git" -ArgumentList @("-C", $repoRoot, "cat-file", "-e", "HEAD:scripts/remote-runner.sh")

$runId = "$ExperimentId-$shortCommit"
$remoteRun = "$remoteBase/runs/$runId"
$remotePreflight = "grep -qx 'xh-202628-execution-mirror-v1' '$remoteBase/.xh-202628-execution-mirror' && test ! -e '$remoteBase/incoming/$runId.tar' && test ! -e '$remoteBase/incoming/$runId.stage.sh' && test ! -e '$remoteRun' && printf 'ready\n'"
$preflightOutput = Get-NativeText -FilePath "ssh" -ArgumentList @("-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR", $sshTarget, $remotePreflight)
if ($preflightOutput -ne "ready") {
    throw "Remote mirror preflight did not return the expected marker."
}
$remoteUsage = Get-NativeText -FilePath "ssh" -ArgumentList @("-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR", $sshTarget, "du -sk -- '$remoteBase'")
if ($remoteUsage -notmatch '^([0-9]+)\s') {
    throw "Could not parse remote storage usage: $remoteUsage"
}
$usedKiB = [int64]$Matches[1]
$maxKiB = [int64]$maxStorageGiB * 1024 * 1024
if ($usedKiB -ge $maxKiB) {
    throw "Remote mirror already uses $usedKiB KiB, at or above the approved limit of $maxKiB KiB."
}

$localResultDir = Join-Path $localResultParent $runId
if (Test-Path -LiteralPath $localResultDir) {
    throw "Local result directory already exists: $localResultDir"
}

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("xh-202628-" + [guid]::NewGuid().ToString("N"))
$bundlePath = Join-Path $tempRoot "$runId.tar"
$stageScript = Join-Path $repoRoot "scripts/remote-stage.sh"
$runnerExitCode = 0

New-Item -ItemType Directory -Path $tempRoot | Out-Null
try {
    Invoke-NativeChecked -FilePath "git" -ArgumentList @("-C", $repoRoot, "archive", "--format=tar", "--output=$bundlePath", "HEAD")
    $bundleKiB = [int64][Math]::Ceiling((Get-Item -LiteralPath $bundlePath).Length / 1KB)
    if (($usedKiB + $bundleKiB) -ge $maxKiB) {
        throw "The source archive would exceed the approved remote storage limit."
    }

    $remoteArchive = "${sshTarget}:${remoteBase}/incoming/$runId.tar"
    $remoteStage = "${sshTarget}:${remoteBase}/incoming/$runId.stage.sh"
    Invoke-NativeChecked -FilePath "scp" -ArgumentList @("-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR", "--", $bundlePath, $remoteArchive)
    Invoke-NativeChecked -FilePath "scp" -ArgumentList @("-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR", "--", $stageScript, $remoteStage)

    $stageCommand = "bash '$remoteBase/incoming/$runId.stage.sh' '$remoteBase' '$runId' '$commit'"
    Invoke-NativeChecked -FilePath "ssh" -ArgumentList @("-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR", $sshTarget, $stageCommand)

    $gpuCsv = $requestedGpuIds -join ','
    $runnerCommand = "bash '$remoteRun/source/scripts/remote-runner.sh' '$remoteBase' '$runId' '$entryPointNormalized' '$commit' '$gpuCsv' '$maxUtil' '$maxMemoryMiB' '$maxRunSeconds' '$maxParallelRuns' '$cudaHome'"
    & ssh -o ClearAllForwardings=yes -o LogLevel=ERROR $sshTarget $runnerCommand
    $runnerExitCode = $LASTEXITCODE

    New-Item -ItemType Directory -Force -Path $localResultParent | Out-Null
    Invoke-NativeChecked -FilePath "scp" -ArgumentList @("-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR", "-r", "${sshTarget}:${remoteRun}/results", $localResultDir)
} finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

Write-Host "Remote run: $runId"
Write-Host "Commit: $commit"
Write-Host "Local raw results: $localResultDir"
if ($runnerExitCode -ne 0) {
    throw "Remote entrypoint failed with exit code $runnerExitCode. Results were retrieved for diagnosis."
}
