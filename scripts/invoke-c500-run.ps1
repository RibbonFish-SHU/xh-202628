[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^exp-[0-9]{8}-[0-9]{3}$')]
    [string]$ExperimentId,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^remote-jobs/[A-Za-z0-9._/-]+[.]sh$')]
    [string]$EntryPoint,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$CandidateCommit,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$BaselineCommit,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$WorkflowCommit,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._/-]+$')]
    [string]$SubmissionSource,

    [ValidateRange(1, 99)]
    [int]$Attempt = 1,

    [ValidatePattern('^exp-[0-9]{8}-[0-9]{3}-[0-9a-f]{12}-a[0-9]{2}$')]
    [string]$RetrieveExistingRun,

    [ValidatePattern('^exp-[0-9]{8}-[0-9]{3}-[0-9a-f]{12}-a[0-9]{2}$')]
    [string]$ResumeStagedRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not [string]::IsNullOrWhiteSpace($RetrieveExistingRun) -and
    -not [string]::IsNullOrWhiteSpace($ResumeStagedRun)) {
    throw "RetrieveExistingRun and ResumeStagedRun are mutually exclusive."
}

function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    $stderrPath = Join-Path ([IO.Path]::GetTempPath()) ("xh-c500-stderr-" + ([guid]::NewGuid()).ToString("N"))
    try {
        $previousErrorAction = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $output = & $FilePath @ArgumentList 2> $stderrPath
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousErrorAction
        }
        $stdout = (@($output | ForEach-Object { [string]$_ }) -join "`n")
        $stderr = if (Test-Path -LiteralPath $stderrPath) {
            Get-Content -Raw -LiteralPath $stderrPath
        } else { "" }
        return [pscustomobject]@{
            ExitCode = $exitCode
            Stdout = (([string]$stdout) -replace '^\s+|\s+$', '')
            Stderr = (([string]$stderr) -replace '^\s+|\s+$', '')
        }
    } catch {
        throw "Native capture failed for '$FilePath': $($_.Exception.Message) $($_.InvocationInfo.PositionMessage)"
    } finally {
        Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    $result = Invoke-NativeCapture -FilePath $FilePath -ArgumentList $ArgumentList
    if ($result.ExitCode -ne 0) {
        $detail = @($result.Stdout, $result.Stderr) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        throw "$FilePath failed with exit code $($result.ExitCode): $($detail -join [Environment]::NewLine)"
    }
    return $result.Stdout
}

function Convert-ToWslDrivePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [IO.Path]::GetFullPath($Path)
    if ($fullPath -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "C500 transfer only supports absolute Windows drive paths: $fullPath"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $remainder = $Matches[2].Replace('\', '/')
    return "/mnt/$drive/$remainder"
}

function Assert-SafeRepoRelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $normalized = $Path.Replace('\', '/')
    $parts = @($normalized.Split('/'))
    if ($normalized.StartsWith('/') -or $parts.Count -eq 0 -or
        @($parts | Where-Object { $_ -in @('', '.', '..') }).Count -gt 0) {
        throw "Unsafe repository-relative path: $Path"
    }
    return $normalized
}

function Assert-ExactCommit {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $resolved = Invoke-NativeChecked -FilePath "git" -ArgumentList @(
        "-C", $Repository, "rev-parse", "$Commit`^{commit}"
    )
    if ($resolved -ne $Commit) {
        throw "$Label must be an exact full commit ID: $Commit"
    }
}

function Get-RegularGitBlob {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    $record = Invoke-NativeChecked -FilePath "git" -ArgumentList @(
        "-C", $Repository, "ls-tree", $Commit, "--", $RelativePath
    )
    if ($record -notmatch '^100(?:644|755) blob ([0-9a-f]{40})\s') {
        throw "Expected a regular committed file at $Commit`:$RelativePath"
    }
    return $Matches[1]
}

function Get-ArchiveContentSha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Archive,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$ExtractionRoot
    )

    New-Item -ItemType Directory -Path $ExtractionRoot | Out-Null
    [void](Invoke-NativeChecked -FilePath "tar.exe" -ArgumentList @(
        "-xf", $Archive, "-C", $ExtractionRoot
    ))
    $contentPath = Join-Path $ExtractionRoot $RelativePath
    if (-not (Test-Path -LiteralPath $contentPath -PathType Leaf)) {
        throw "Git archive did not contain the expected source: $RelativePath"
    }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $contentPath).Hash.ToLowerInvariant()
}

function Convert-ToFiniteDouble {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Label,
        [switch]$Positive
    )

    try {
        $number = [double]$Value
    } catch {
        throw "C500 evidence field is not numeric: $Label"
    }
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number) -or
        ($Positive -and $number -le 0.0)) {
        throw "C500 evidence field is not a valid finite number: $Label"
    }
    return $number
}

function Get-FinitePositiveArray {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][int]$ExpectedCount,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $items = @($Value)
    if ($items.Count -ne $ExpectedCount) {
        throw "C500 evidence array has the wrong length: $Label"
    }
    return @(
        for ($index = 0; $index -lt $items.Count; $index++) {
            Convert-ToFiniteDouble -Value $items[$index] `
                -Label "$Label[$index]" -Positive
        }
    )
}

function Assert-CloseNumber {
    param(
        [Parameter(Mandatory = $true)][double]$Actual,
        [Parameter(Mandatory = $true)][double]$Expected,
        [Parameter(Mandatory = $true)][double]$AbsoluteTolerance,
        [Parameter(Mandatory = $true)][double]$RelativeTolerance,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $tolerance = [Math]::Max($AbsoluteTolerance, [Math]::Abs($Expected) * $RelativeTolerance)
    if ([Math]::Abs($Actual - $Expected) -gt $tolerance) {
        throw "C500 evidence value is inconsistent with raw samples: $Label"
    }
}

function Assert-C500Status {
    param(
        [Parameter(Mandatory = $true)]$Status,
        [Parameter(Mandatory = $true)][string]$ExpectedRun,
        [Parameter(Mandatory = $true)][string]$ExpectedCandidate,
        [Parameter(Mandatory = $true)][string]$ExpectedBaseline,
        [Parameter(Mandatory = $true)][string]$ExpectedWorkflow,
        [Parameter(Mandatory = $true)][string]$ExpectedEntryPoint,
        [Parameter(Mandatory = $true)][string]$ExpectedSource,
        [Parameter(Mandatory = $true)][string]$ExpectedCandidateSourceSha,
        [Parameter(Mandatory = $true)][string]$ExpectedBaselineSourceSha,
        [Parameter(Mandatory = $true)][string]$ExpectedWorkflowArchiveSha,
        [Parameter(Mandatory = $true)][string]$ExpectedCandidateArchiveSha,
        [Parameter(Mandatory = $true)][string]$ExpectedBaselineArchiveSha,
        [Parameter(Mandatory = $true)][string]$ExpectedStageSha,
        [switch]$AllowStaged
    )

    $expectedValues = @{
        run_id = $ExpectedRun
        commit = $ExpectedCandidate
        baseline_commit = $ExpectedBaseline
        workflow_commit = $ExpectedWorkflow
        entrypoint = $ExpectedEntryPoint
        submission_source = $ExpectedSource
        candidate_source_sha256 = $ExpectedCandidateSourceSha
        baseline_source_sha256 = $ExpectedBaselineSourceSha
        workflow_archive_sha256 = $ExpectedWorkflowArchiveSha
        candidate_archive_sha256 = $ExpectedCandidateArchiveSha
        baseline_archive_sha256 = $ExpectedBaselineArchiveSha
        stage_sha256 = $ExpectedStageSha
        device_class = "c500-local"
    }
    if ([int]$Status.schema_version -ne 2) {
        throw "C500 status has an unsupported schema version."
    }
    foreach ($key in $expectedValues.Keys) {
        if ([string]$Status.$key -ne [string]$expectedValues[$key]) {
            throw "C500 status field '$key' does not match the requested run."
        }
    }
    if (-not $ExpectedRun.StartsWith("$ExperimentId-$($ExpectedCandidate.Substring(0, 12))-") -or
        $ExpectedRun -notmatch '-a[0-9]{2}$') {
        throw "C500 run ID does not bind the requested experiment and candidate commit."
    }

    $state = [string]$Status.state
    $exitCode = [int]$Status.exit_code
    $validTerminal = switch ($state) {
        "succeeded" { $exitCode -eq 0 }
        "failed" { $exitCode -ne 0 }
        "preflight-failed" { $exitCode -eq 125 }
        "staging-failed" { $exitCode -ne 0 }
        "interrupted" { $exitCode -in @(129, 130, 143) }
        "staged" { $AllowStaged -and $exitCode -eq 0 }
        default { $false }
    }
    if (-not $validTerminal) {
        throw "C500 status state/exit_code is not valid for this operation: $state/$exitCode"
    }
    return $Status
}

function Get-VerifiedLocalStatus {
    param(
        [Parameter(Mandatory = $true)][string]$ResultDirectory,
        [Parameter(Mandatory = $true)][hashtable]$Expected
    )

    $statusPath = Join-Path $ResultDirectory "status.json"
    if (-not (Test-Path -LiteralPath $statusPath -PathType Leaf)) {
        throw "Retrieved C500 results do not contain status.json: $ResultDirectory"
    }
    $status = Get-Content -Raw -Encoding utf8 -LiteralPath $statusPath | ConvertFrom-Json
    [void](Assert-C500Status -Status $status `
        -ExpectedRun $Expected.Run `
        -ExpectedCandidate $Expected.Candidate `
        -ExpectedBaseline $Expected.Baseline `
        -ExpectedWorkflow $Expected.Workflow `
        -ExpectedEntryPoint $Expected.EntryPoint `
        -ExpectedSource $Expected.Source `
        -ExpectedCandidateSourceSha $Expected.CandidateSourceSha `
        -ExpectedBaselineSourceSha $Expected.BaselineSourceSha `
        -ExpectedWorkflowArchiveSha $Expected.WorkflowArchiveSha `
        -ExpectedCandidateArchiveSha $Expected.CandidateArchiveSha `
        -ExpectedBaselineArchiveSha $Expected.BaselineArchiveSha `
        -ExpectedStageSha $Expected.StageSha)

    if ([string]$status.state -eq "succeeded") {
        $requiredFiles = @(
            "environment.txt", "correctness.log", "regression.log",
            "build-candidate.log", "build-baseline.log", "warmup-baseline.log",
            "warmup-candidate.log", "benchmark-baseline-a.log",
            "benchmark-candidate-a.log", "benchmark-candidate-b.log",
            "benchmark-baseline-b.log", "paired-benchmark.json",
            "mx-smi-before.txt", "mx-smi-before.json.txt", "mx-smi-process-before.txt",
            "mx-smi-after.txt", "mx-smi-after.json.txt", "mx-smi-process-after.txt",
            "result-manifest.sha256"
        )
        foreach ($name in $requiredFiles) {
            if (-not (Test-Path -LiteralPath (Join-Path $ResultDirectory $name) -PathType Leaf)) {
                throw "Successful C500 run is missing required evidence: $name"
            }
        }

        $paired = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $ResultDirectory "paired-benchmark.json") | ConvertFrom-Json
        if ([int]$paired.schema_version -ne 1 -or
            [string]$paired.device_class -ne "c500-local" -or
            [string]$paired.benchmark_design -ne "ABBA" -or
            [string]$paired.candidate_commit -ne $Expected.Candidate -or
            [string]$paired.baseline_commit -ne $Expected.Baseline -or
            [string]$paired.submission_source -ne $Expected.Source -or
            [string]$paired.candidate_source_sha256 -ne $Expected.CandidateSourceSha -or
            [string]$paired.baseline_source_sha256 -ne $Expected.BaselineSourceSha) {
            throw "paired-benchmark.json does not match the requested C500 run."
        }
        $caseNames = @($paired.cases.psobject.Properties.Name | Sort-Object)
        $expectedCases = @("decode-down", "decode-gate-up", "prefill-down", "prefill-gate-up")
        if (($caseNames -join "|") -ne ($expectedCases -join "|")) {
            throw "paired-benchmark.json does not contain the four required public cases."
        }
        foreach ($caseName in $expectedCases) {
            $casePayload = $paired.cases.psobject.Properties[$caseName].Value
            $baselineMedians = @(Get-FinitePositiveArray `
                -Value $casePayload.baseline_ms_ab -ExpectedCount 2 `
                -Label "$caseName.baseline_ms_ab")
            $candidateMedians = @(Get-FinitePositiveArray `
                -Value $casePayload.candidate_ms_ab -ExpectedCount 2 `
                -Label "$caseName.candidate_ms_ab")
            $baselineRawRuns = @($casePayload.baseline_raw_samples_ms_ab)
            $candidateRawRuns = @($casePayload.candidate_raw_samples_ms_ab)
            if ($baselineRawRuns.Count -ne 2 -or $candidateRawRuns.Count -ne 2) {
                throw "paired-benchmark.json must contain two raw AB runs per arm: $caseName"
            }
            for ($runIndex = 0; $runIndex -lt 2; $runIndex++) {
                $baselineSamples = @(Get-FinitePositiveArray `
                    -Value $baselineRawRuns[$runIndex] -ExpectedCount 5 `
                    -Label "$caseName.baseline_raw_samples_ms_ab[$runIndex]")
                $candidateSamples = @(Get-FinitePositiveArray `
                    -Value $candidateRawRuns[$runIndex] -ExpectedCount 5 `
                    -Label "$caseName.candidate_raw_samples_ms_ab[$runIndex]")
                $baselineSorted = @($baselineSamples | Sort-Object)
                $candidateSorted = @($candidateSamples | Sort-Object)
                Assert-CloseNumber -Actual $baselineMedians[$runIndex] `
                    -Expected $baselineSorted[2] -AbsoluteTolerance 0.0005 `
                    -RelativeTolerance 0.0 `
                    -Label "$caseName baseline run $runIndex median"
                Assert-CloseNumber -Actual $candidateMedians[$runIndex] `
                    -Expected $candidateSorted[2] -AbsoluteTolerance 0.0005 `
                    -RelativeTolerance 0.0 `
                    -Label "$caseName candidate run $runIndex median"
            }

            $baselineSummary = Convert-ToFiniteDouble `
                -Value $casePayload.baseline_median_ms `
                -Label "$caseName.baseline_median_ms" -Positive
            $candidateSummary = Convert-ToFiniteDouble `
                -Value $casePayload.candidate_median_ms `
                -Label "$caseName.candidate_median_ms" -Positive
            $speedup = Convert-ToFiniteDouble -Value $casePayload.speedup `
                -Label "$caseName.speedup" -Positive
            $candidateDelta = Convert-ToFiniteDouble `
                -Value $casePayload.candidate_delta_percent `
                -Label "$caseName.candidate_delta_percent"
            $baselineDrift = Convert-ToFiniteDouble `
                -Value $casePayload.baseline_drift_percent `
                -Label "$caseName.baseline_drift_percent"
            $candidateDrift = Convert-ToFiniteDouble `
                -Value $casePayload.candidate_drift_percent `
                -Label "$caseName.candidate_drift_percent"
            $calculatedBaseline = ($baselineMedians[0] + $baselineMedians[1]) / 2.0
            $calculatedCandidate = ($candidateMedians[0] + $candidateMedians[1]) / 2.0
            Assert-CloseNumber -Actual $baselineSummary -Expected $calculatedBaseline `
                -AbsoluteTolerance 1.0e-9 -RelativeTolerance 1.0e-12 `
                -Label "$caseName combined baseline median"
            Assert-CloseNumber -Actual $candidateSummary -Expected $calculatedCandidate `
                -AbsoluteTolerance 1.0e-9 -RelativeTolerance 1.0e-12 `
                -Label "$caseName combined candidate median"
            Assert-CloseNumber -Actual $speedup `
                -Expected ($calculatedBaseline / $calculatedCandidate) `
                -AbsoluteTolerance 1.0e-9 -RelativeTolerance 1.0e-12 `
                -Label "$caseName speedup"
            Assert-CloseNumber -Actual $candidateDelta `
                -Expected (($calculatedCandidate / $calculatedBaseline - 1.0) * 100.0) `
                -AbsoluteTolerance 1.0e-8 -RelativeTolerance 1.0e-12 `
                -Label "$caseName candidate delta"
            Assert-CloseNumber -Actual $baselineDrift `
                -Expected (($baselineMedians[1] / $baselineMedians[0] - 1.0) * 100.0) `
                -AbsoluteTolerance 1.0e-8 -RelativeTolerance 1.0e-12 `
                -Label "$caseName baseline drift"
            Assert-CloseNumber -Actual $candidateDrift `
                -Expected (($candidateMedians[1] / $candidateMedians[0] - 1.0) * 100.0) `
                -AbsoluteTolerance 1.0e-8 -RelativeTolerance 1.0e-12 `
                -Label "$caseName candidate drift"
        }

        $manifestPath = Join-Path $ResultDirectory "result-manifest.sha256"
        $manifestLines = @(Get-Content -Encoding utf8 -LiteralPath $manifestPath)
        if ($manifestLines.Count -eq 0) {
            throw "C500 result manifest is empty."
        }
        $manifestNames = @{}
        foreach ($line in $manifestLines) {
            if ($line -notmatch '^([0-9a-f]{64})  ([A-Za-z0-9._-]+)$') {
                throw "C500 result manifest contains an unsafe or malformed entry: $line"
            }
            $expectedDigest = $Matches[1]
            $manifestName = $Matches[2]
            if ($manifestNames.ContainsKey($manifestName)) {
                throw "C500 result manifest contains a duplicate entry: $manifestName"
            }
            $manifestNames[$manifestName] = $true
            $manifestFile = Join-Path $ResultDirectory $manifestName
            if (-not (Test-Path -LiteralPath $manifestFile -PathType Leaf)) {
                throw "C500 result manifest references a missing file: $manifestName"
            }
            $actualDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestFile).Hash.ToLowerInvariant()
            if ($actualDigest -ne $expectedDigest) {
                throw "C500 result manifest digest mismatch: $manifestName"
            }
        }
        $actualEvidenceNames = @(
            Get-ChildItem -LiteralPath $ResultDirectory -File |
                Where-Object { $_.Name -notin @("status.json", "result-manifest.sha256") } |
                ForEach-Object { $_.Name } |
                Sort-Object
        )
        $manifestEvidenceNames = @($manifestNames.Keys | Sort-Object)
        if (($actualEvidenceNames -join "|") -ne ($manifestEvidenceNames -join "|")) {
            throw "C500 result manifest does not cover the exact retrieved evidence file set."
        }
    }
    return $status
}

function Receive-C500Results {
    param(
        [Parameter(Mandatory = $true)][string]$RemoteResults,
        [Parameter(Mandatory = $true)][string]$FinalDirectory,
        [Parameter(Mandatory = $true)][hashtable]$Expected,
        [Parameter(Mandatory = $true)][string[]]$ScpArguments,
        [Parameter(Mandatory = $true)][string]$SshAlias,
        [int]$TransportExitCode = -1
    )

    if (Test-Path -LiteralPath $FinalDirectory) {
        $existingStatus = Get-VerifiedLocalStatus `
            -ResultDirectory $FinalDirectory -Expected $Expected
        if ($TransportExitCode -ge 0 -and
            [int]$existingStatus.exit_code -ne $TransportExitCode) {
            if ($TransportExitCode -eq 255) {
                Write-Warning (
                    "SSH transport ended with 255; using the already verified terminal " +
                    "C500 status $($existingStatus.state)/$($existingStatus.exit_code)."
                )
            } else {
                throw "C500 runner exit code $TransportExitCode differs from the verified existing status exit code $($existingStatus.exit_code)."
            }
        }
        Write-Warning "Using already verified local C500 results: $FinalDirectory"
        return $existingStatus
    }
    $parent = Split-Path -Parent $FinalDirectory
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $partialDirectory = "$FinalDirectory.partial-$(([guid]::NewGuid()).ToString('N'))"
    $wslPartial = Convert-ToWslDrivePath -Path $partialDirectory
    try {
        [void](Invoke-NativeChecked -FilePath "wsl.exe" -ArgumentList @(
            $ScpArguments + @("-r", "$SshAlias`:$RemoteResults", $wslPartial)
        ))
        $status = Get-VerifiedLocalStatus -ResultDirectory $partialDirectory -Expected $Expected
        if ($TransportExitCode -ge 0 -and [int]$status.exit_code -ne $TransportExitCode) {
            if ($TransportExitCode -eq 255) {
                Write-Warning (
                    "SSH transport ended with 255; using fully retrieved and verified " +
                    "terminal C500 status $($status.state)/$($status.exit_code)."
                )
            } else {
                throw "C500 runner exit code $TransportExitCode differs from status.json exit code $($status.exit_code)."
            }
        }
        [IO.Directory]::Move($partialDirectory, $FinalDirectory)
        return $status
    } catch {
        $detail = $_.Exception.Message
        throw "C500 result retrieval or verification failed. Any partial evidence remains at '$partialDirectory'. $detail"
    }
}

function Get-VerifiedStagedRemoteStatus {
    param(
        [Parameter(Mandatory = $true)][string]$RemoteBase,
        [Parameter(Mandatory = $true)][string]$RemoteRun,
        [Parameter(Mandatory = $true)][string]$MarkerFilename,
        [Parameter(Mandatory = $true)][string]$MarkerValue,
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][hashtable]$Expected,
        [Parameter(Mandatory = $true)][string[]]$SshArguments
    )

    $command =
        "grep -qx '$MarkerValue' '$RemoteBase/$MarkerFilename' && " +
        "test -d '$RemoteRun' && test ! -L '$RemoteRun' && " +
        "test -f '$RemoteRun/results/status.json' && " +
        "test ! -L '$RemoteRun/results/status.json' && " +
        "test ! -e '$RemoteRun/results/started-at.txt' && " +
        "test ! -e '$RemoteRun/results/finished-at.txt' && " +
        "test ! -e '$RemoteRun/.runner-claim' && " +
        "test ! -e '$RemoteBase/locks/c500-run-slot.lock' && " +
        "! pgrep -f -- 'c500-runner[.]sh.*$RunId' >/dev/null && " +
        "cat '$RemoteRun/results/status.json'"
    $captured = Invoke-NativeCapture -FilePath "wsl.exe" -ArgumentList @(
        $SshArguments + $command
    )
    if ($captured.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($captured.Stdout)) {
        $detail = @($captured.Stdout, $captured.Stderr) |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        throw "Remote C500 run is not safely resumable from staged state: $($detail -join [Environment]::NewLine)"
    }
    $status = $captured.Stdout | ConvertFrom-Json
    [void](Assert-C500Status -Status $status `
        -ExpectedRun $Expected.Run `
        -ExpectedCandidate $Expected.Candidate `
        -ExpectedBaseline $Expected.Baseline `
        -ExpectedWorkflow $Expected.Workflow `
        -ExpectedEntryPoint $Expected.EntryPoint `
        -ExpectedSource $Expected.Source `
        -ExpectedCandidateSourceSha $Expected.CandidateSourceSha `
        -ExpectedBaselineSourceSha $Expected.BaselineSourceSha `
        -ExpectedWorkflowArchiveSha $Expected.WorkflowArchiveSha `
        -ExpectedCandidateArchiveSha $Expected.CandidateArchiveSha `
        -ExpectedBaselineArchiveSha $Expected.BaselineArchiveSha `
        -ExpectedStageSha $Expected.StageSha `
        -AllowStaged)
    if ([string]$status.state -ne "staged") {
        throw "Remote C500 run is not in staged state."
    }
    return $status
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$gitRoot = Invoke-NativeChecked -FilePath "git" -ArgumentList @(
    "-C", $repoRoot, "rev-parse", "--show-toplevel"
)
if ([IO.Path]::GetFullPath($gitRoot) -ne $repoRoot) {
    throw "Script must run from the xh-202628-agent Git repository."
}
$commonGitDir = Invoke-NativeChecked -FilePath "git" -ArgumentList @(
    "-C", $repoRoot, "rev-parse", "--path-format=absolute", "--git-common-dir"
)
$primaryRepoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $commonGitDir))
$primaryGitRoot = Invoke-NativeChecked -FilePath "git" -ArgumentList @(
    "-C", $primaryRepoRoot, "rev-parse", "--show-toplevel"
)
if ([IO.Path]::GetFullPath($primaryGitRoot) -ne $primaryRepoRoot) {
    throw "Could not resolve the primary worktree from the Git common directory."
}

$statePath = [IO.Path]::GetFullPath((Join-Path $repoRoot "state/c500-execution.json"))
$config = Get-Content -Raw -Encoding utf8 -LiteralPath $statePath | ConvertFrom-Json
if ($config.schema_version -ne 1) {
    throw "Unsupported C500 state schema: $($config.schema_version)"
}
if (-not $config.directory_creation.authorized -or -not $config.directory_creation.created) {
    throw "The C500 execution mirror has not been authorized and recorded as created."
}
if (-not $config.execution.enabled) {
    throw "C500 execution is disabled in state/c500-execution.json."
}
if ([string]$config.transport.kind -ne "wsl-openssh-control-connection" -or
    -not $config.transport.batch_mode_required -or
    [string]$config.transport.credential_storage -ne "none") {
    throw "C500 transport must use the credential-free WSL control connection policy."
}

$sshAlias = [string]$config.transport.ssh_alias
$remoteBase = [string]$config.base_directory
$markerFilename = [string]$config.marker.filename
$markerValue = [string]$config.marker.value
$macaPath = [string]$config.toolchain.maca_path
if ($sshAlias -ne "xh-c500") {
    throw "Only the repository-approved SSH alias xh-c500 is allowed."
}
if ($remoteBase -ne "/root/xh-202628-agent") {
    throw "Remote base differs from the approved C500 execution mirror."
}
if ($markerFilename -ne ".xh-202628-c500-execution-mirror" -or
    $markerValue -ne "xh-202628-c500-execution-mirror-v1") {
    throw "Unexpected C500 execution marker configuration."
}
if ($macaPath -ne "/opt/maca") {
    throw "The C500 runner requires the verified MACA installation at /opt/maca."
}
if ([int]$config.execution.device_id -ne 0 -or [int]$config.execution.max_parallel_runs -ne 1) {
    throw "C500 execution must remain single-device and single-run."
}
if ([string]$config.execution.controller_role -ne "main-agent-only" -or
    [string]$config.execution.source_projection -ne "workflow-commit-plus-source-and-template-job" -or
    [string]$config.execution.candidate_environment -ne "env-i-explicit-allowlist" -or
    [string]$config.execution.candidate_privilege -ne "root" -or
    [string]$config.execution.filesystem_isolation -ne "none" -or
    [int]$config.execution.status_schema_version -ne 2 -or
    -not $config.execution.evidence_manifest_required) {
    throw "C500 trusted-control and evidence policy differs from the approved workflow."
}

$maxUtil = [int]$config.execution.max_start_utilization_percent
$maxVram = [int]$config.execution.max_start_vram_used_mib
$maxRunSeconds = [int]$config.execution.max_run_seconds
$maxStorageGiB = [int]$config.execution.max_remote_storage_gib
$expectedCompute = [int]$config.execution.expected_compute_quota_percent
$expectedVram = [int]$config.execution.expected_vram_quota_mib
if ($maxUtil -lt 0 -or $maxUtil -gt 100 -or $maxVram -lt 0 -or
    $maxRunSeconds -le 0 -or $maxStorageGiB -le 0 -or
    $expectedCompute -le 0 -or $expectedCompute -gt 100 -or $expectedVram -le 0) {
    throw "C500 execution thresholds are invalid."
}

$entryPointNormalized = Assert-SafeRepoRelativePath -Path $EntryPoint
$sourceNormalized = Assert-SafeRepoRelativePath -Path $SubmissionSource
$status = Invoke-NativeChecked -FilePath "git" -ArgumentList @(
    "-C", $repoRoot, "status", "--porcelain=v1", "--untracked-files=all"
)
if (-not [string]::IsNullOrWhiteSpace($status)) {
    throw "C500 control requires a completely clean, committed worktree.`n$status"
}

Assert-ExactCommit -Repository $repoRoot -Commit $CandidateCommit -Label "CandidateCommit"
Assert-ExactCommit -Repository $repoRoot -Commit $BaselineCommit -Label "BaselineCommit"
Assert-ExactCommit -Repository $repoRoot -Commit $WorkflowCommit -Label "WorkflowCommit"
$workflowReservation = Invoke-NativeChecked -FilePath "git" -ArgumentList @(
    "-C", $repoRoot, "rev-parse", "--verify", "refs/xh-202628/experiments/$ExperimentId"
)
if ($workflowReservation -ne $WorkflowCommit) {
    throw "WorkflowCommit does not match the immutable experiment reservation."
}
$baselineReservation = Invoke-NativeChecked -FilePath "git" -ArgumentList @(
    "-C", $repoRoot, "rev-parse", "--verify", "refs/xh-202628/baselines/$ExperimentId"
)
if ($baselineReservation -ne $BaselineCommit) {
    throw "BaselineCommit does not match the immutable experiment baseline reservation."
}
$ancestor = Invoke-NativeCapture -FilePath "git" -ArgumentList @(
    "-C", $repoRoot, "merge-base", "--is-ancestor", $WorkflowCommit, $CandidateCommit
)
if ($ancestor.ExitCode -ne 0) {
    throw "CandidateCommit must descend from the explicit WorkflowCommit."
}

$trustedControlPaths = @("scripts/invoke-c500-run.ps1", "scripts/c500-stage.sh")
foreach ($path in $trustedControlPaths) {
    [void](Get-RegularGitBlob -Repository $repoRoot -Commit $WorkflowCommit -RelativePath $path)
    $comparison = Invoke-NativeCapture -FilePath "git" -ArgumentList @(
        "-C", $repoRoot, "diff", "--quiet", $WorkflowCommit, "HEAD", "--", $path
    )
    if ($comparison.ExitCode -ne 0) {
        throw "Local C500 control file differs from WorkflowCommit: $path"
    }
}

$workflowRequired = @(
    "templates/remote-job.sh",
    "scripts/c500-runner.sh",
    "scripts/run-c500-fused-moe-paired.sh",
    "scripts/summarize-c500-abba.py",
    "operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
)
foreach ($required in $workflowRequired) {
    [void](Get-RegularGitBlob -Repository $repoRoot -Commit $WorkflowCommit -RelativePath $required)
}
$workflowTree = Invoke-NativeChecked -FilePath "git" -ArgumentList @(
    "-C", $repoRoot, "ls-tree", "-r", $WorkflowCommit
)
$unsupportedWorkflowEntries = @($workflowTree -split "`n" | Where-Object {
    $_ -and $_ -notmatch '^100(?:644|755) blob [0-9a-f]{40}\s'
})
if ($unsupportedWorkflowEntries.Count -gt 0) {
    throw "WorkflowCommit contains a symlink, submodule or unsupported Git entry."
}
foreach ($candidateRequired in @($entryPointNormalized, $sourceNormalized)) {
    [void](Get-RegularGitBlob -Repository $repoRoot -Commit $CandidateCommit -RelativePath $candidateRequired)
}
$candidateEntryBlob = Get-RegularGitBlob -Repository $repoRoot -Commit $CandidateCommit -RelativePath $entryPointNormalized
$trustedEntryBlob = Get-RegularGitBlob -Repository $repoRoot -Commit $WorkflowCommit -RelativePath "templates/remote-job.sh"
if ($candidateEntryBlob -ne $trustedEntryBlob) {
    throw "Candidate entrypoint must be an exact copy of WorkflowCommit:templates/remote-job.sh."
}
$baselineSourceBlob = Get-RegularGitBlob -Repository $repoRoot -Commit $BaselineCommit -RelativePath $sourceNormalized
$workflowSourceBlob = Get-RegularGitBlob -Repository $repoRoot -Commit $WorkflowCommit -RelativePath $sourceNormalized
if ($baselineSourceBlob -ne $workflowSourceBlob) {
    throw "WorkflowCommit and BaselineCommit must begin from the same submission-source blob."
}

$sshOptions = @(
    "ssh", "-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
    "-o", "LogLevel=ERROR", $sshAlias
)
$scpOptions = @(
    "scp", "-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
    "-o", "LogLevel=ERROR"
)
[void](Invoke-NativeChecked -FilePath "wsl.exe" -ArgumentList @($sshOptions + "true"))

$shortCommit = $CandidateCommit.Substring(0, 12)
$attemptSuffix = "a{0:D2}" -f $Attempt
$runId = "$ExperimentId-$shortCommit-$attemptSuffix"
if (-not [string]::IsNullOrWhiteSpace($RetrieveExistingRun)) {
    $runId = $RetrieveExistingRun
} elseif (-not [string]::IsNullOrWhiteSpace($ResumeStagedRun)) {
    $runId = $ResumeStagedRun
}
$remoteRun = "$remoteBase/runs/$runId"
$localResultParent = Join-Path $primaryRepoRoot "artifacts/raw/c500-runs"
$localResultDir = Join-Path $localResultParent $runId

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("xh-c500-bundle-" + ([guid]::NewGuid()).ToString("N"))
$workflowBundle = Join-Path $tempRoot "$runId.workflow.tar"
$candidateBundle = Join-Path $tempRoot "$runId.candidate-source.tar"
$baselineBundle = Join-Path $tempRoot "$runId.baseline-source.tar"
$workflowControlRoot = Join-Path $tempRoot "workflow-control"
$stageScript = Join-Path $workflowControlRoot "scripts/c500-stage.sh"
New-Item -ItemType Directory -Path $tempRoot | Out-Null

try {
    [void](Invoke-NativeChecked -FilePath "git" -ArgumentList @(
        "-C", $repoRoot, "archive", "--format=tar", "--output=$workflowBundle", $WorkflowCommit
    ))
    [void](Invoke-NativeChecked -FilePath "git" -ArgumentList @(
        "-C", $repoRoot, "archive", "--format=tar", "--output=$candidateBundle",
        $CandidateCommit, "--", $sourceNormalized, $entryPointNormalized
    ))
    [void](Invoke-NativeChecked -FilePath "git" -ArgumentList @(
        "-C", $repoRoot, "archive", "--format=tar", "--output=$baselineBundle",
        $BaselineCommit, "--", $sourceNormalized
    ))

    $workflowArchiveSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $workflowBundle).Hash.ToLowerInvariant()
    $candidateArchiveSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidateBundle).Hash.ToLowerInvariant()
    $baselineArchiveSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $baselineBundle).Hash.ToLowerInvariant()
    $stageSha = Get-ArchiveContentSha256 `
        -Archive $workflowBundle -RelativePath "scripts/c500-stage.sh" `
        -ExtractionRoot $workflowControlRoot
    $candidateSourceSha = Get-ArchiveContentSha256 `
        -Archive $candidateBundle -RelativePath $sourceNormalized `
        -ExtractionRoot (Join-Path $tempRoot "candidate-content")
    $baselineSourceSha = Get-ArchiveContentSha256 `
        -Archive $baselineBundle -RelativePath $sourceNormalized `
        -ExtractionRoot (Join-Path $tempRoot "baseline-content")

    $expected = @{
        Run = $runId
        Candidate = $CandidateCommit
        Baseline = $BaselineCommit
        Workflow = $WorkflowCommit
        EntryPoint = $entryPointNormalized
        Source = $sourceNormalized
        CandidateSourceSha = $candidateSourceSha
        BaselineSourceSha = $baselineSourceSha
        WorkflowArchiveSha = $workflowArchiveSha
        CandidateArchiveSha = $candidateArchiveSha
        BaselineArchiveSha = $baselineArchiveSha
        StageSha = $stageSha
    }

    $expectedRunPrefix = "$ExperimentId-$shortCommit-"
    if ((-not [string]::IsNullOrWhiteSpace($RetrieveExistingRun) -or
         -not [string]::IsNullOrWhiteSpace($ResumeStagedRun)) -and
        -not $runId.StartsWith($expectedRunPrefix)) {
        throw "Existing run does not match ExperimentId and CandidateCommit."
    }

    if (-not [string]::IsNullOrWhiteSpace($RetrieveExistingRun)) {
        $remoteStatusCommand =
            "grep -qx '$markerValue' '$remoteBase/$markerFilename' && " +
            "test -d '$remoteRun' && test ! -L '$remoteRun' && " +
            "test -f '$remoteRun/results/status.json' && " +
            "test ! -L '$remoteRun/results/status.json' && " +
            "cat '$remoteRun/results/status.json'"
        $remoteStatusText = Invoke-NativeChecked -FilePath "wsl.exe" -ArgumentList @(
            $sshOptions + $remoteStatusCommand
        )
        $remoteStatus = $remoteStatusText | ConvertFrom-Json
        [void](Assert-C500Status -Status $remoteStatus `
            -ExpectedRun $expected.Run `
            -ExpectedCandidate $expected.Candidate `
            -ExpectedBaseline $expected.Baseline `
            -ExpectedWorkflow $expected.Workflow `
            -ExpectedEntryPoint $expected.EntryPoint `
            -ExpectedSource $expected.Source `
            -ExpectedCandidateSourceSha $expected.CandidateSourceSha `
            -ExpectedBaselineSourceSha $expected.BaselineSourceSha `
            -ExpectedWorkflowArchiveSha $expected.WorkflowArchiveSha `
            -ExpectedCandidateArchiveSha $expected.CandidateArchiveSha `
            -ExpectedBaselineArchiveSha $expected.BaselineArchiveSha `
            -ExpectedStageSha $expected.StageSha)
        $localStatus = Receive-C500Results `
            -RemoteResults "$remoteRun/results" `
            -FinalDirectory $localResultDir `
            -Expected $expected `
            -ScpArguments $scpOptions `
            -SshAlias $sshAlias
        if ([string]$localStatus.state -ne [string]$remoteStatus.state -or
            [int]$localStatus.exit_code -ne [int]$remoteStatus.exit_code) {
            throw "Retrieved C500 status differs from the verified remote terminal status."
        }
        Write-Host "Recovered C500 run: $runId"
        Write-Host "Candidate commit: $CandidateCommit"
        Write-Host "Paired baseline commit: $BaselineCommit"
        Write-Host "Trusted workflow commit: $WorkflowCommit"
        Write-Host "Local raw results: $localResultDir"
        return
    }

    $incomingWorkflow = "$remoteBase/incoming/$runId.workflow.tar"
    $incomingCandidate = "$remoteBase/incoming/$runId.candidate-source.tar"
    $incomingBaseline = "$remoteBase/incoming/$runId.baseline-source.tar"
    $incomingStage = "$remoteBase/incoming/$runId.stage.sh"
    $resumeReady = $false
    if (-not [string]::IsNullOrWhiteSpace($ResumeStagedRun)) {
        [void](Get-VerifiedStagedRemoteStatus `
            -RemoteBase $remoteBase -RemoteRun $remoteRun `
            -MarkerFilename $markerFilename -MarkerValue $markerValue `
            -RunId $runId -Expected $expected -SshArguments $sshOptions)
        $resumeReady = $true
        Write-Warning "Resuming fully verified staged C500 run: $runId"
    }

    if (-not $resumeReady) {
        $remotePreflight =
            "grep -qx '$markerValue' '$remoteBase/$markerFilename' && " +
            "test ! -e '$incomingWorkflow' && test ! -e '$incomingCandidate' && " +
            "test ! -e '$incomingBaseline' && test ! -e '$incomingStage' && " +
            "test ! -e '$remoteRun' && test ! -e '$remoteBase/runs/.staging-$runId' && printf 'ready\n'"
        $preflightOutput = Invoke-NativeChecked -FilePath "wsl.exe" -ArgumentList @(
            $sshOptions + $remotePreflight
        )
        if ($preflightOutput -ne "ready") {
            throw "C500 mirror preflight did not return the expected marker."
        }
        $remoteUsage = Invoke-NativeChecked -FilePath "wsl.exe" -ArgumentList @(
            $sshOptions + "du -sk -- '$remoteBase'"
        )
        if ($remoteUsage -notmatch '^([0-9]+)\s') {
            throw "Could not parse C500 mirror storage usage: $remoteUsage"
        }
        $usedKiB = [int64]$Matches[1]
        $maxKiB = [int64]$maxStorageGiB * 1024 * 1024
        $bundleKiB = [int64][Math]::Ceiling(
            ((Get-Item -LiteralPath $workflowBundle).Length * 3 +
             (Get-Item -LiteralPath $candidateBundle).Length * 2 +
             (Get-Item -LiteralPath $baselineBundle).Length * 2) / 1KB
        )
        if (($usedKiB + $bundleKiB) -ge $maxKiB) {
            throw "The staged C500 source, baseline and retained archives would exceed the approved storage limit."
        }
        if (Test-Path -LiteralPath $localResultDir) {
            throw "Local result directory already exists: $localResultDir"
        }

        $uploads = @(
            @($workflowBundle, $incomingWorkflow),
            @($candidateBundle, $incomingCandidate),
            @($baselineBundle, $incomingBaseline),
            @($stageScript, $incomingStage)
        )
        foreach ($upload in $uploads) {
            $wslSource = Convert-ToWslDrivePath -Path $upload[0]
            [void](Invoke-NativeChecked -FilePath "wsl.exe" -ArgumentList @(
                $scpOptions + @("--", $wslSource, "$sshAlias`:$($upload[1])")
            ))
        }

        $stageCommand =
            "bash '$incomingStage' '$remoteBase' '$runId' '$CandidateCommit' '$BaselineCommit' " +
            "'$WorkflowCommit' '$sourceNormalized' '$entryPointNormalized' '$markerValue' " +
            "'$workflowArchiveSha' '$candidateArchiveSha' '$baselineArchiveSha' '$stageSha' " +
            "'$candidateSourceSha' '$baselineSourceSha'"
        $stageResult = Invoke-NativeCapture -FilePath "wsl.exe" -ArgumentList @(
            $sshOptions + $stageCommand
        )
        if ($stageResult.ExitCode -ne 0) {
            if ($stageResult.ExitCode -eq 255) {
                try {
                    [void](Get-VerifiedStagedRemoteStatus `
                        -RemoteBase $remoteBase -RemoteRun $remoteRun `
                        -MarkerFilename $markerFilename -MarkerValue $markerValue `
                        -RunId $runId -Expected $expected -SshArguments $sshOptions)
                    $resumeReady = $true
                    Write-Warning (
                        "C500 staging transport ended with 255, but the remote run is " +
                        "fully verified as staged; continuing with the runner."
                    )
                } catch {
                    Write-Warning $_.Exception.Message
                }
            }
            if (-not $resumeReady) {
                $terminalRecovered = $false
                try {
                    [void](Receive-C500Results `
                        -RemoteResults "$remoteRun/results" `
                        -FinalDirectory $localResultDir `
                        -Expected $expected `
                        -ScpArguments $scpOptions `
                        -SshAlias $sshAlias `
                        -TransportExitCode $stageResult.ExitCode)
                    $terminalRecovered = $true
                } catch {
                    Write-Warning $_.Exception.Message
                }
                if ($terminalRecovered) {
                    throw "C500 staging reached a verified terminal failure with exit code $($stageResult.ExitCode)."
                }
                if ($stageResult.ExitCode -eq 255) {
                    throw "C500 staging transport is uncertain. Retry with ResumeStagedRun for a verified staged run or RetrieveExistingRun for a verified terminal run."
                }
                throw "C500 staging failed with exit code $($stageResult.ExitCode)."
            }
        }
    }

    $runnerCommand =
        "bash '$remoteRun/source/scripts/c500-runner.sh' '$remoteBase' '$runId' " +
        "'$entryPointNormalized' '$CandidateCommit' '$BaselineCommit' '$WorkflowCommit' " +
        "'$maxUtil' '$maxVram' '$maxRunSeconds' '$expectedCompute' '$expectedVram' '$macaPath'"
    $runnerResult = Invoke-NativeCapture -FilePath "wsl.exe" -ArgumentList @(
        $sshOptions + $runnerCommand
    )
    $localStatus = Receive-C500Results `
        -RemoteResults "$remoteRun/results" `
        -FinalDirectory $localResultDir `
        -Expected $expected `
        -ScpArguments $scpOptions `
        -SshAlias $sshAlias `
        -TransportExitCode $runnerResult.ExitCode

    Write-Host "C500 run: $runId"
    Write-Host "Candidate commit: $CandidateCommit"
    Write-Host "Paired baseline commit: $BaselineCommit"
    Write-Host "Trusted workflow commit: $WorkflowCommit"
    Write-Host "Local raw results: $localResultDir"
    if ([int]$localStatus.exit_code -ne 0) {
        throw "C500 entrypoint reached $($localStatus.state) with exit code $($localStatus.exit_code). Results were retrieved for diagnosis."
    }
} finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
