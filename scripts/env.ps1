# Dot-source this file to use the optional project-local toolchain in your shell.
$revoProjectRoot=Split-Path -Parent $PSScriptRoot
if(Test-Path "$revoProjectRoot\.tools\cargo\bin\cargo.exe") {
    $env:CARGO_HOME="$revoProjectRoot\.tools\cargo"
    $env:RUSTUP_HOME="$revoProjectRoot\.tools\rustup"
    $env:PATH="$revoProjectRoot\.tools\cargo\bin;$env:PATH"
}
if(Test-Path "$revoProjectRoot\.tools\cuda\nvidia\cuda_nvrtc\bin") {
    $env:PATH="$revoProjectRoot\.tools\cuda\nvidia\cuda_nvrtc\bin;$env:PATH"
}
