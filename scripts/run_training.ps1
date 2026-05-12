# =============================================================================
#  run_training.ps1
#  End-to-end single-case ELGIN training pipeline.
#
#  Pipeline: extract -> mesh -> train -> rollout -> animate.
#  Output layout :
#
#    experiments\elgin_case03\
#      datasets\          case_single.npz  +  mesh_graph.npz
#      models\            best.pt, config.json, stage_*.pt
#      results\
#        logs\            extract / mesh / train / rollout / animation logs
#        rollouts\        rollout.npz, gt.npz, clinical_metrics.json
#        animations\      compare.mp4, fluid_speed_particles.mp4
#        metrics\         case_metrics.json
#
#  Usage examples:
#    cd $PSScriptRoot\..
#    .\scripts\run_training.ps1                         # full pipeline, paper-spec 300 epochs
#    .\scripts\run_training.ps1 -Epochs 100             # quick smoke-test
#    .\scripts\run_training.ps1 -SkipExtract            # reuse existing case_single.npz
#    .\scripts\run_training.ps1 -SkipExtract -SkipMesh -SkipTrain   # rollout + animate only
# =============================================================================

[CmdletBinding()]
param(
    # --- Epoch budget --------------------------------------------------------
    # Paper Table III recipe: 300 total epochs split 60/60/120/60 across
    # Stages 1/2/3/4.  Reduce to 100-150 for a quick smoke-test.
    [int]    $Epochs       = 300,

    [int]    $BatchSize    = 4,
    # Paper Table III: d_h = 64, K_E = K_L = 4.
    [int]    $HiddenSize   = 64,
    [int]    $MpSteps      = 4,
    [int]    $NParticles   = 1000,
    [int]    $NSteps       = 255,
    [double] $TStart       = 2.0,
    [double] $TEnd         = 28.0,
    [double] $DtKeep       = 0.1,

    # --- BPTT rollout fine-tuning (Stage 4) ----------------------------------
    # Paper Table III: BPTT unroll = 5.  Each BPTT step is sequential so
    # bptt_steps dominates Stage 4 cost; reduce to 3 for a faster run.
    [int]    $BpttSteps    = 5,

    # Weight of BPTT rollout loss vs one-step supervised loss in Stage 4.
    [double] $BpttWeight   = 0.5,

    # --- Stage 4 only (resume from existing Stage-3 checkpoint) --------------
    # Pass -Stage4Only to skip Stages 1-3 and jump straight to Stage 4 BPTT
    # using the best.pt already saved in the model directory.
    [switch] $Stage4Only,

    [string] $CaseName     = "dentalRoom2D",
    [string] $Device       = "cuda",
    [switch] $SkipExtract,
    [switch] $SkipMesh,
    [switch] $SkipTrain,
    [switch] $SkipRollout,
    [switch] $SkipAnimate,
    [switch] $Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = "$PSScriptRoot\.."
Set-Location $ROOT

# =============================================================================
#  Path definitions
# =============================================================================
$EXP        = Join-Path $ROOT "experiments"
$OF_CASE    = Join-Path $ROOT "openfoam\$CaseName"

$EXP_DIR    = Join-Path $EXP  "elgin_case03"
$DATA_DIR   = Join-Path $EXP_DIR "datasets"
$MODEL_DIR  = Join-Path $EXP_DIR "models"
$LOGS_DIR   = Join-Path $EXP_DIR "results\logs"
$ROLL_DIR   = Join-Path $EXP_DIR "results\rollouts"
$ANIM_DIR   = Join-Path $EXP_DIR "results\animations"
$MET_DIR    = Join-Path $EXP_DIR "results\metrics"

@(
    $DATA_DIR, $MODEL_DIR, $LOGS_DIR, $ROLL_DIR, $ANIM_DIR, $MET_DIR
) | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

# =============================================================================
#  Helpers
# =============================================================================
function Write-Banner {
    param([string]$Text)
    $line = "=" * 70
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
    Write-Host ""
}

function Write-Status {
    param([string]$Text)
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts]  $Text" -ForegroundColor Yellow
}

function Invoke-Step {
    param(
        [string]   $Label,
        [string[]] $Cmd,
        [string]   $Log,
        [switch]   $AllowFail
    )
    Write-Status "START  $Label"
    $t0 = Get-Date
    & $Cmd[0] $Cmd[1..($Cmd.Length - 1)] 2>&1 | Tee-Object -FilePath $Log
    $exitCode = $LASTEXITCODE
    $elapsed  = (Get-Date) - $t0
    $hms      = "{0:hh\:mm\:ss}" -f $elapsed
    if ($exitCode -ne 0 -and -not $AllowFail) {
        Write-Host "  [FAILED]  $Label  exit=$exitCode  elapsed=$hms" -ForegroundColor Red
        Write-Host "  Log: $Log" -ForegroundColor Red
        exit $exitCode
    }
    Write-Status "DONE   $Label  ($hms)"
    return $elapsed
}

# =============================================================================
#  Sanity checks
# =============================================================================
Write-Banner "ELGIN single-case run  ($CaseName, $NParticles particles)"

if (-not (Test-Path $OF_CASE)) {
    Write-Host "  [ERROR] OpenFOAM case not found: $OF_CASE" -ForegroundColor Red
    Write-Host "  Place the foam-extend 4.1 reactingParcelFoam case in openfoam\$CaseName" -ForegroundColor Yellow
    Write-Host "  (see openfoam\README.md for the expected setup)." -ForegroundColor Yellow
    exit 1
}

try {
    python -c "import torch; g=torch.cuda.get_device_name(0); m=torch.cuda.get_device_properties(0).total_memory//1024**3; print(f'  GPU: {g}  ({m} GB VRAM)')"
} catch {
    Write-Host "  [WARNING] No CUDA GPU detected -- using CPU (slower)." -ForegroundColor Yellow
}

if ($Clean) {
    Write-Status "Cleaning previous outputs in $EXP_DIR"
    Remove-Item -Recurse -Force "$DATA_DIR\*"  -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$MODEL_DIR\*" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$ROLL_DIR\*"  -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$ANIM_DIR\*"  -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$MET_DIR\*"   -ErrorAction SilentlyContinue
}

$wallStart = Get-Date
$summary   = [System.Collections.Generic.List[string]]::new()

# =============================================================================
#  STEP 1 - Extract field data (1000 unique parcels with origId tracking)
# =============================================================================
$NPZ_PATH = Join-Path $DATA_DIR "case_single.npz"

if ($SkipExtract -and (Test-Path $NPZ_PATH)) {
    Write-Status "[skip] $($NPZ_PATH | Split-Path -Leaf) already exists."
} else {
    $t = Invoke-Step "Extract  $CaseName" @(
        "python", "-u", "elgin\data\extract_fields.py",
        "--case_dir",    $OF_CASE,
        "--output",      $NPZ_PATH,
        "--t_start",     "$TStart",
        "--t_end",       "$TEnd",
        "--dt_keep",     "$DtKeep",
        "--n_particles", "$NParticles"
    ) "$LOGS_DIR\extract.log"
    $summary.Add(("Extract                 : {0:hh\:mm\:ss}" -f $t))
}

# =============================================================================
#  STEP 2 - Build mesh graph
# =============================================================================
$MESH_PATH = Join-Path $DATA_DIR "mesh_graph.npz"

if ($SkipMesh -and (Test-Path $MESH_PATH)) {
    Write-Status "[skip] $($MESH_PATH | Split-Path -Leaf) already exists."
} else {
    $t = Invoke-Step "Mesh graph" @(
        "python", "-u", "elgin\data\mesh_to_graph.py",
        "--case_dir", $OF_CASE,
        "--output",   $MESH_PATH
    ) "$LOGS_DIR\mesh.log"
    $summary.Add(("Mesh graph              : {0:hh\:mm\:ss}" -f $t))
}

# =============================================================================
#  STEP 3 - Train ELGIN  (re-uses train_single.py with paper-spec defaults)
# =============================================================================
if (-not $SkipTrain) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

    $trainCmd = @(
        "python", "-u", "elgin\train_single.py",
        "--case_dir",    $OF_CASE,
        "--out_dir",     $DATA_DIR,
        "--model_dir",   $MODEL_DIR,
        "--device",      $Device,
        "--n_particles", "$NParticles",
        "--epochs",      "$Epochs",
        "--batch_size",  "$BatchSize",
        "--hidden_size", "$HiddenSize",
        "--mp_steps",    "$MpSteps",
        "--t_start",     "$TStart",
        "--t_end",       "$TEnd",
        "--dt_keep",     "$DtKeep",
        # -- BPTT rollout fine-tuning (Stage 4) ---------------------------
        # --freeze_fluid_stage4: feed GT fluid at every BPTT step, matching
        #   the --freeze_fluid flag used at rollout time.
        "--freeze_fluid_stage4",
        "--bptt_steps",          "$BpttSteps",
        "--bptt_weight",         "$BpttWeight",
        "--bptt_rollout_noise",  "0.01",
        # -- Stage 3 PDE lambda weights -----------------------------------
        # Reduced from defaults (mom=0.05, cont=0.10, turb=0.02) to keep
        # Stage 3 as a gentle regulariser rather than an aggressive override.
        "--lambda_mom",  "0.001",
        "--lambda_cont", "0.01",
        "--lambda_turb", "0.001",
        "--skip_extract",
        "--skip_mesh"
    )
    if ($Stage4Only) { $trainCmd += @("--start_stage", "4") }

    $t = Invoke-Step "Train ELGIN ($Epochs epochs)" $trainCmd "$LOGS_DIR\train.log"
    $summary.Add(("Train                   : {0:hh\:mm\:ss}" -f $t))
}

# =============================================================================
#  STEP 4 - Rollout  (writes rollout.npz AND ID-matched gt.npz)
# =============================================================================
if (-not $SkipRollout) {
    $t = Invoke-Step "Rollout $CaseName ($NSteps steps)" @(
        "python", "-u", "elgin\rollout.py",
        "--model_dir",   $MODEL_DIR,
        "--mesh",        $MESH_PATH,
        "--ic_file",     $NPZ_PATH,
        "--n_particles", "$NParticles",
        "--n_steps",     "$NSteps",
        "--output",      $ROLL_DIR,
        "--device",      $Device,
        "--freeze_fluid"
    ) "$LOGS_DIR\rollout.log"
    $summary.Add(("Rollout                 : {0:hh\:mm\:ss}" -f $t))

    $met = Join-Path $ROLL_DIR "clinical_metrics.json"
    if (Test-Path $met) {
        Copy-Item $met (Join-Path $MET_DIR "case_metrics.json") -Force
    }
}

# =============================================================================
#  STEP 5 - Animations
# =============================================================================
if (-not $SkipAnimate) {
    $rolloutNpz = Join-Path $ROLL_DIR "rollout.npz"
    $gtNpz      = Join-Path $ROLL_DIR "gt.npz"
    if ((Test-Path $rolloutNpz) -and (Test-Path $gtNpz)) {
        # Single panel: fluid speed + GNN particles
        $t = Invoke-Step "Fluid+particle animation (single)" @(
            "python", "-u", "elgin\animate_fluid_particles.py",
            "--rollout", $rolloutNpz,
            "--output",  (Join-Path $ANIM_DIR "fluid_speed_particles.mp4"),
            "--mode",    "speed",
            "--fps",     "10"
        ) "$LOGS_DIR\fluid_anim.log" -AllowFail
        $summary.Add(("Fluid anim (single)     : {0:hh\:mm\:ss}" -f $t))

        # Two-panel: ELGIN vs GT side-by-side, fluid speed background
        $t = Invoke-Step "Fluid+particle animation (compare)" @(
            "python", "-u", "elgin\animate_fluid_particles.py",
            "--rollout", $rolloutNpz,
            "--gt",      $gtNpz,
            "--output",  (Join-Path $ANIM_DIR "fluid_speed_compare.mp4"),
            "--mode",    "speed",
            "--fps",     "10"
        ) "$LOGS_DIR\fluid_anim_compare.log" -AllowFail
        $summary.Add(("Fluid anim (compare)    : {0:hh\:mm\:ss}" -f $t))
    } else {
        Write-Host "  [skip] No rollout.npz / gt.npz found for fluid animation." -ForegroundColor Yellow
    }
}

# =============================================================================
#  Summary
# =============================================================================
$wallElapsed = (Get-Date) - $wallStart
Write-Banner "Summary"
foreach ($l in $summary) { Write-Host "  $l" }
Write-Host ""
Write-Host ("  Total wall time         : {0:hh\:mm\:ss}" -f $wallElapsed) -ForegroundColor Green
Write-Host ""
Write-Host "  Outputs:"
Write-Host "    Data        : $NPZ_PATH"
Write-Host "    Model       : $(Join-Path $MODEL_DIR 'best.pt')"
Write-Host "    Rollout     : $(Join-Path $ROLL_DIR 'rollout.npz')"
Write-Host "    GT (matched): $(Join-Path $ROLL_DIR 'gt.npz')"
Write-Host "    Animation   : $(Join-Path $ANIM_DIR 'fluid_speed_compare.mp4')"
Write-Host ""
