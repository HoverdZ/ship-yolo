param(
    [Parameter(Mandatory = $true)]
    [string]$Source,

    [Parameter(Mandatory = $false)]
    [string]$DestinationRoot = "D:\遥感船舶检测论文\03_正式实验结果",

    [Parameter(Mandatory = $false)]
    [string]$Version,

    [Parameter(Mandatory = $false)]
    [switch]$FromZip,

    [Parameter(Mandatory = $false)]
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Arguments = @()
if ($FromZip) {
    $Collector = Join-Path $RepositoryRoot "tools\windows_collect_from_zip.py"
} else {
    $Collector = Join-Path $RepositoryRoot "tools\windows_collect_from_drive.py"
}
$Arguments += $Collector
$Arguments += $Source
$Arguments += $DestinationRoot
if ($Version) {
    $Arguments += "--version"
    $Arguments += $Version
}
if ($Resume) {
    $Arguments += "--resume"
}

& python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Collection failed with exit code $LASTEXITCODE."
}

Write-Host "Collection completed. Run tools\verify_collected_artifacts.py on the version directory."
