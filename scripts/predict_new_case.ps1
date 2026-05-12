# =============================================================================
#  predict_new_case.ps1
#  Run ELGIN inference on an UNKNOWN situation using the trained model.
#
#  Inputs accepted
#  ---------------
#    -InputPath   Path to EITHER:
#                   (a) a pre-extracted .npz file  (fastest — no OF required)
#                   (b) a raw OpenFOAM case directory (auto-extracted first)
#
#  Outputs (in -OutputDir)
#  -----------------------
#    rollout.npz           raw trajectory data (fluid + particles)
#    clinical_metrics.json exposure / deposition fractions
#    fluid_particles.mp4   animated air speed + aerosol prediction
#    compare.mp4           side-by-side ELGIN vs GT  (only if -GtNpz given)
#    logs\                 extraction + rollout + animation logs
#
#  Usage examples
#  --------------
#    # Predict from an already-extracted NPZ
#    cd $PSScriptRoot\..
#    .\scripts\predict_new_case.ps1 `
#        -InputPath "experiments\elgin_case03\datasets\case_single.npz" `
#        -OutputDir "predictions\sweep_case_03"
#
#    # Predict straight from a raw OpenFOAM directory
#    .\scripts\predict_new_case.ps1 `
#        -InputPath "D:\openfoam\Sweep_Case_03" `
#        -OutputDir "predictions\sweep_case_03"
#
#    # Compare against ground truth (e.g. validate on a known case)
#    .\scripts\predict_new_case.ps1 `
#        -InputPath "experiments\elgin_case03\datasets\case_single.npz" `
#        -GtNpz     "experiments\elgin_case03\datasets\case_single.npz" `
#        -OutputDir "predictions\sweep_case_03_validation"
# =============================================================================

[CmdletBinding()]
param(
    # ── Required ───────────────────────────────────────────────────────────
    [Parameter(Mandatory=$true)]
    [string] $InputPath,               # .npz file  OR  OpenFOAM case directory

    # ── Optional: output location ──────────────────────────────────────────
    [string] $OutputDir    = "predictions\new_case",

    # ── Optional: model / mesh ─────────────────────────────────────────────
    [string] $ModelDir     = "experiments\elgin_case03\models",
    [string] $MeshPath     = "experiments\elgin_case03\datasets\mesh_graph.npz",

    # ── Optional: comparison ground truth ─────────────────────────────────
    [string] $GtNpz        = "",        # leave blank for prediction-only mode

    # ── Simulation settings ────────────────────────────────────────────────
    [int]    $NParticles   = 1000,
    [int]    $NSteps       = 255,       # ~25.5 s at dt = 0.1 s
    [string] $Device       = "cuda",

    # ── Extraction settings (only used if InputPath is an OF directory) ────
    [float]  $TStart       = 2.0,
    [float]  $TEnd         = 28.0,
    [float]  $DtKeep       = 0.1,

    # ── Animation settings ─────────────────────────────────────────────────
    [int]    $Fps          = 10,
    [switch] $SkipAnimate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = "$PSScriptRoot\.."
Set-Location $ROOT

# =============================================================================
#  Banner
# =============================================================================
$line = "=" * 68
Write-Host ""
Write-Host $line                        -ForegroundColor Cyan
Write-Host "  ELGIN  Prediction for Unknown Case" -ForegroundColor Cyan
Write-Host $line                        -ForegroundColor Cyan
Write-Host ""
Write-Host "  Input      : $InputPath"
Write-Host "  Model      : $ModelDir"
Write-Host "  Output     : $OutputDir"
Write-Host "  Device     : $Device"
Write-Host ""

# =============================================================================
#  Validate model exists
# =============================================================================
$bestPt = Join-Path $ROOT "$ModelDir\best.pt"
if (-not (Test-Path $bestPt)) {
    Write-Host "[ERROR] Trained model not found: $bestPt" -ForegroundColor Red
    Write-Host "  Train first with:  .\scripts\run_training.ps1" -ForegroundColor Yellow
    exit 1
}

# =============================================================================
#  Build Python command
# =============================================================================
$pyArgs = @(
    "python", "-u", "elgin\predict_new_case.py",
    "--input",       $InputPath,
    "--model_dir",   $ModelDir,
    "--mesh",        $MeshPath,
    "--output_dir",  $OutputDir,
    "--n_particles", "$NParticles",
    "--n_steps",     "$NSteps",
    "--device",      $Device,
    "--t_start",     "$TStart",
    "--t_end",       "$TEnd",
    "--dt_keep",     "$DtKeep",
    "--fps",         "$Fps"
)

if ($GtNpz -ne "") {
    $pyArgs += @("--gt_npz", $GtNpz)
}

if ($SkipAnimate) {
    $pyArgs += "--skip_animate"
}

# =============================================================================
#  Run prediction
# =============================================================================
$t0 = Get-Date
& $pyArgs[0] $pyArgs[1..($pyArgs.Length - 1)]
$exitCode = $LASTEXITCODE
$elapsed  = "{0:hh\:mm\:ss}" -f ((Get-Date) - $t0)

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "[FAILED]  Prediction exited with code $exitCode  (elapsed: $elapsed)" -ForegroundColor Red
    exit $exitCode
}

# =============================================================================
#  Summary
# =============================================================================
Write-Host ""
Write-Host $line                        -ForegroundColor Green
Write-Host "  Prediction complete  ($elapsed)" -ForegroundColor Green
Write-Host $line                        -ForegroundColor Green
Write-Host ""
Write-Host "  Outputs saved to: $(Join-Path $ROOT $OutputDir)"
Write-Host ""
Write-Host "  Key files:"
Write-Host "    rollout.npz            — raw trajectory data"
Write-Host "    clinical_metrics.json  — exposure & deposition fractions"
Write-Host "    fluid_particles.mp4    — air velocity + aerosol animation"
if ($GtNpz -ne "") {
    Write-Host "    compare.mp4            — GNN vs ground truth comparison"
}
Write-Host ""
Write-Host "  To re-animate with different settings:"
Write-Host "    python elgin\animate_fluid_particles.py ``"
Write-Host "        --rollout $(Join-Path $OutputDir 'rollout.npz') ``"
Write-Host "        --output  my_animation.mp4 --mode speed --fps 10"
Write-Host ""
