# Project-local tools are optional. A normal Rust/CUDA installation also works.
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CargoArgs)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\env.ps1"
Push-Location $projectRoot
try {
    if(!$CargoArgs) {$CargoArgs=@('run','--release')}
    & cargo @CargoArgs
    exit $LASTEXITCODE
} finally {Pop-Location}
