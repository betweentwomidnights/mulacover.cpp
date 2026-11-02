param([switch]$Shared,[ValidateSet('cpu','cuda')][string]$Backend='cpu',
  [string]$DevelopmentGgml)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$cmake='C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
if(Get-Command cmake -ErrorAction SilentlyContinue){$cmake=(Get-Command cmake).Source}
$build=if($Shared){'build-windows-dll'}else{'build-windows-cpu'}
if($DevelopmentGgml){$build="build-windows-development-$Backend";if($Shared){$build+='-dll'}}
$flags=@("-DMULACOVER_CUDA=$(@{cpu='OFF';cuda='ON'}[$Backend])")
if($Backend -eq 'cuda'){$flags+='-DCMAKE_CUDA_ARCHITECTURES=120a-real'}
if($DevelopmentGgml){$flags+="-DMULACOVER_GGML_SOURCE_DIR=$DevelopmentGgml"}
else{$flags+='-DMULACOVER_GGML_SOURCE_DIR=';if($Backend -eq 'cuda'){$build='build-windows-cuda';if($Shared){$build+='-dll'}}}
$static=if($Shared){'OFF'}else{'ON'}
& $cmake -S $projectRoot -B (Join-Path $projectRoot $build) -G 'Visual Studio 17 2022' -A x64 `
  -DMULACOVER_USE_INSTALLED_GGML=OFF -DGGML_NATIVE=OFF -DHEARTCODEC_BUILD_TOOLS=ON -DHEARTCODEC_BUILD_TESTS=ON "-DMULACOVER_STATIC=$static" @flags
if($LASTEXITCODE){throw 'CMake configuration failed'}
& $cmake --build (Join-Path $projectRoot $build) --config Release --parallel 4 -- /verbosity:minimal /nologo
if($LASTEXITCODE){throw 'Build failed'}
$ctest=Join-Path (Split-Path -Parent $cmake) 'ctest.exe'
& $ctest --test-dir (Join-Path $projectRoot $build) -C Release --output-on-failure
if($LASTEXITCODE){throw 'C ABI test failed'}
