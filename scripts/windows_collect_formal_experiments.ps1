[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [string]$Target = "D:\遥感船舶检测论文\formal_ablation_v1",
    [string]$Python = "python",
    [switch]$SkipSummary
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptRoot
$Collector = Join-Path $ScriptRoot "windows_collect_formal_experiments.py"

$Arguments = @(
    $Collector,
    "--source", (Resolve-Path -LiteralPath $Source).Path,
    "--target", $Target,
    "--repo", $RepoRoot
)
if ($SkipSummary) {
    $Arguments += "--skip-summary"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Formal experiment collection failed with exit code $LASTEXITCODE."
}
