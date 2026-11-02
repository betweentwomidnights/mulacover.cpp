param([switch]$Shared)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$cmake='C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
if(Get-Command cmake -ErrorAction SilentlyContinue){$cmake=(Get-Command cmake).Source}
$build=if($Shared){'build-windows-dll'}else{'build-windows-cpu'}
$static=if($Shared){'OFF'}else{'ON'}
& $cmake -S $projectRoot -B (Join-Path $projectRoot $build) -G 'Visual Studio 17 2022' -A x64 `
  -DMULACOVER_USE_INSTALLED_GGML=OFF -DMULACOVER_CUDA=OFF -DGGML_NATIVE=OFF "-DMULACOVER_STATIC=$static"
if($LASTEXITCODE){throw 'CMake configuration failed'}
& $cmake --build (Join-Path $projectRoot $build) --config Release --parallel 4 -- /verbosity:minimal /nologo
if($LASTEXITCODE){throw 'Build failed'}
$ctest=Join-Path (Split-Path -Parent $cmake) 'ctest.exe'
& $ctest --test-dir (Join-Path $projectRoot $build) -C Release --output-on-failure
if($LASTEXITCODE){throw 'C ABI test failed'}
