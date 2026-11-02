param([string]$Audio,[ValidatePattern('^[A-Za-z0-9_-]+$')][string]$CaseName=('cover-'+[guid]::NewGuid().ToString('N').Substring(0,8)),
  [ValidateSet('instrumental','vocal','combined')][string]$MelodyRole='instrumental',
  [ValidateRange(1,5000)][int]$Sixteenths=64,[ValidateRange(1,8192)][int]$Frames=100,
  [ValidateRange(1,8192)][int]$CacheCapacity=256,
  [switch]$FullSong,
  [ValidateSet('native','torch-cuda')][string]$Rng='native',
  [ValidateRange(0,1048576)][int]$RngCudaBlocks=0,[ValidateRange(0,128)][int]$RngDevice=0,
  [string]$Lyrics='[Instrumental]',
  [string]$Tags='topic:[Instrumental]; genre:[Classical]; instrument:[Violin, Piano]; mood:[dramatic]',
  [string]$TranscriberCli,[string]$TranscriberModel,[string]$GeneratorModel,[string]$DevelopmentGgml)
$ErrorActionPreference='Stop'
if($Rng -eq 'native' -and ($RngCudaBlocks -ne 0 -or $RngDevice -ne 0)){throw 'CUDA RNG options require torch-cuda mode'}
if($FullSong -and ($PSBoundParameters.ContainsKey('Frames') -or $PSBoundParameters.ContainsKey('Sixteenths'))){throw 'FullSong derives duration from the source; omit Frames and Sixteenths'}
if(-not $FullSong -and $Frames -gt $CacheCapacity){throw 'Frames must fit within CacheCapacity, with additional room for the prompt'}
$projectRoot=Split-Path -Parent $PSScriptRoot
$siblings=Split-Path -Parent $projectRoot
$python=Join-Path $siblings 'heartcodec.cpp/.venv/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $python)){$python='python'}
$selected=Get-Content (Join-Path $projectRoot 'docs/validation/transcription-cover-input.json') -Raw|ConvertFrom-Json
if(-not $Audio){$Audio=$selected.source_path}
if(-not $TranscriberCli){$TranscriberCli=Join-Path $siblings 'yue2.cpp/build-ggml027-cuda/bin/Release/yue2-transcribe.exe'}
if(-not $TranscriberModel){$TranscriberModel=Join-Path $siblings 'yue2.cpp/models/sheetsage2-mert2-f16.gguf'}
if(-not $GeneratorModel){$GeneratorModel=Join-Path $projectRoot 'models/mulacover-q4_k_m.gguf'}
if(-not $DevelopmentGgml){$DevelopmentGgml=Join-Path $siblings '_ggml_reviews/cuda-f32-source-b77'}
$localRoot=Join-Path $projectRoot "refdata/covers/$CaseName"
if(Test-Path -LiteralPath $localRoot){throw 'Use a fresh CaseName; existing outputs are preserved'}
$transcription=Join-Path $localRoot 'transcription'
$transcribeArguments=@('--audio',$Audio,'--cli',$TranscriberCli,'--model',$TranscriberModel,'--output',$transcription,
  '--melody-role',$MelodyRole,'--sixteenths',$Sixteenths,'--device','cuda')
if($FullSong){$transcribeArguments+='--full-song'}
if([IO.Path]::GetFullPath($Audio) -eq [IO.Path]::GetFullPath($selected.source_path)){
  $transcribeArguments+=@('--expected-audio-sha256',$selected.sha256)
}
& $python (Join-Path $projectRoot 'tools/transcribe-cover.py') @transcribeArguments
if($LASTEXITCODE){throw 'Native transcription/MIDI audit failed'}
$source=Get-Content (Join-Path $transcription 'transcription.json') -Raw|ConvertFrom-Json
if($FullSong){
  $Frames=[int][Math]::Ceiling($source.source_audio.frames/$source.source_audio.sample_rate/0.08)
  if($Frames -gt 8192){throw 'Full source exceeds the native generation frame limit'}
}
$request=[ordered]@{lyrics=$Lyrics;tags=$Tags;melody_midi='transcription/excerpt-midi/melody.mid';
  chord_midi='transcription/excerpt-midi/chords.mid';melody_role=$MelodyRole;source_audio_sha256=$source.source_sha256}
$requestFile=Join-Path $localRoot 'request.json'
[IO.File]::WriteAllText($requestFile,($request|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
# CaseName is restricted to letters/digits/hyphen/underscore. Audio and text are
# data files, never interpolated into the remote shell. Only the two MIDIs travel.
$remoteRoot="/home/kev/dev/mulacover.cpp/refdata/covers/$CaseName"
& ssh spark-c7ca "test ! -e $remoteRoot && mkdir -p $remoteRoot/transcription/excerpt-midi"
if($LASTEXITCODE){throw 'Remote case already exists or Spark is unavailable'}
& scp $requestFile "spark-c7ca:$remoteRoot/request.json"
if($LASTEXITCODE){throw 'Request transfer failed'}
& scp (Join-Path $transcription 'excerpt-midi/melody.mid') (Join-Path $transcription 'excerpt-midi/chords.mid') "spark-c7ca:$remoteRoot/transcription/excerpt-midi/"
if($LASTEXITCODE){throw 'MIDI transfer failed'}
$remoteCommand="docker run --rm --gpus all --user 1000:1000 --network none -e USER=heartcodec -e LOGNAME=heartcodec -e HOME=/workspace -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v /home/kev/dev/mulacover.cpp:/workspace -v /home/kev/dev/_heartcodec_refs:/refs:ro -w /workspace heartcodec-reference:20261101 /opt/sa3-venv/bin/python reference/prepare-conditioning.py --upstream /refs/MuLaCover --request refdata/covers/$CaseName/request.json --output refdata/covers/$CaseName/prepared --frames $Frames"
& ssh spark-c7ca $remoteCommand
if($LASTEXITCODE){throw 'Spark Docker conditioning preparation failed'}
& scp -r "spark-c7ca:$remoteRoot/prepared" $localRoot
if($LASTEXITCODE){throw 'Prepared-input transfer failed'}
$preparedManifest=Get-Content (Join-Path $localRoot 'prepared/manifest.json') -Raw|ConvertFrom-Json
$promptFrames=$preparedManifest.preprocessing.'input.tokens'.shape[1]
if($FullSong -and -not $PSBoundParameters.ContainsKey('CacheCapacity')){
  $CacheCapacity=[int]([Math]::Ceiling(($promptFrames+$Frames-1)/256.0)*256)
}
if($promptFrames+$Frames-1 -gt $CacheCapacity){throw 'CacheCapacity must hold the prompt and requested frames'}
& $python (Join-Path $projectRoot 'tools/generate.py') --reference (Join-Path $localRoot 'prepared') `
  --model $GeneratorModel --cli (Join-Path $projectRoot 'build-windows-development-cuda/bin/Release/mulacover-generate.exe') `
  --output (Join-Path $localRoot 'generation') --ggml-source $DevelopmentGgml --frames $Frames --capacity $CacheCapacity `
  --rng $Rng --rng-cuda-blocks $RngCudaBlocks --rng-device $RngDevice `
  --flow-model (Join-Path $siblings 'heartcodec.cpp/models/heartcodec-flow-f32.gguf') `
  --sq-model (Join-Path $siblings 'heartcodec.cpp/models/heartcodec-sq-codec-f32.gguf')
if($LASTEXITCODE){throw 'Native generation/decode failed'}
Write-Output "Audio: $(Join-Path $localRoot 'generation/audio.wav')"
