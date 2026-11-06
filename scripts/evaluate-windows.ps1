param([ValidateSet('cpu','cuda','vulkan')][string]$Backend='cuda',
  [ValidateSet('quantization','parity')][string]$Evaluation='quantization',
  [string]$Model,[int]$Capacity=8192)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
if(-not $Model){$Model=Join-Path $projectRoot 'models/mulacover-q4_k_m.gguf'}
$ggml=Join-Path (Split-Path $projectRoot -Parent) 'heartcodec.cpp/ggml'
$cli=Join-Path $projectRoot "build-windows-$Backend/bin/Release/mulacover-parity.exe"
$python=Join-Path (Split-Path $projectRoot -Parent) 'heartcodec.cpp/.venv/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $python)){$python='python'}
$output=Join-Path $projectRoot "refdata/ggml-029-$([IO.Path]::GetFileNameWithoutExtension($Model))-$Backend-cap$Capacity"
$flags=@()
if($Evaluation -eq 'quantization'){$flags+='--measure-quantization'}
if($Backend -eq 'cuda'){$flags+='--sample-device-memory'}
$env:GGML_CUDA_TF32='0';$env:GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto';$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& $python (Join-Path $projectRoot 'tools/run-parity.py') --reference (Join-Path $projectRoot 'refdata/upstream-f32') `
  --model $Model --cli $cli --output $output --backend $Backend --ggml-source $ggml `
  --capacity $Capacity @flags
if($LASTEXITCODE){throw 'Native evaluation failed; inspect stdout/stderr/report under refdata'}
Write-Output "Report: $output/report.json"
