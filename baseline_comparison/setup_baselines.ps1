# Arnis Pose Baseline Comparison - Environment Setup

Write-Host "Setting up environments for Arnis Baseline Comparison..." -ForegroundColor Cyan

# Define paths
$baseDir = "baseline_comparison"
$models = @("xgboost", "mlp", "capsnet")

foreach ($model in $models) {
    Write-Host "`n--- Setting up $model ---" -ForegroundColor Yellow
    $modelDir = Join-Path $baseDir $model
    $venvDir = Join-Path $modelDir "venv_$model"
    
    if (-not (Test-Path $venvDir)) {
        Write-Host "Creating virtual environment in $venvDir..."
        python -m venv $venvDir
    } else {
        Write-Host "Virtual environment already exists in $venvDir."
    }
    
    $pipPath = Join-Path $venvDir "Scripts\pip.exe"
    $reqPath = Join-Path $modelDir "requirements.txt"
    
    Write-Host "Installing requirements from $reqPath..."
    & $pipPath install -r $reqPath
}

Write-Host "`nSetup Complete!" -ForegroundColor Green
