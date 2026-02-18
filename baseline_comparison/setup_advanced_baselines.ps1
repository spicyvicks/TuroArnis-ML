# PowerShell script to set up advanced baseline environments
# Run this script to create virtual environments and install dependencies

Write-Host "Setting up Advanced Baseline Models..." -ForegroundColor Cyan

# Transformer
Write-Host "`nSetting up Pose Transformer..." -ForegroundColor Yellow
if (-not (Test-Path "baseline_comparison\transformer\venv")) {
    python -m venv baseline_comparison\transformer\venv
    & baseline_comparison\transformer\venv\Scripts\Activate.ps1
    pip install -r baseline_comparison\transformer\requirements.txt
    deactivate
    Write-Host "Transformer environment created!" -ForegroundColor Green
}
else {
    Write-Host "Transformer environment already exists." -ForegroundColor Gray
}

# EdgeConv
Write-Host "`nSetting up EdgeConv..." -ForegroundColor Yellow
if (-not (Test-Path "baseline_comparison\edgeconv\venv")) {
    python -m venv baseline_comparison\edgeconv\venv
    & baseline_comparison\edgeconv\venv\Scripts\Activate.ps1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install torch-geometric
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    pip install -r baseline_comparison\edgeconv\requirements.txt
    deactivate
    Write-Host "EdgeConv environment created!" -ForegroundColor Green
}
else {
    Write-Host "EdgeConv environment already exists." -ForegroundColor Gray
}

# ST-GCN
Write-Host "`nSetting up ST-GCN..." -ForegroundColor Yellow
if (-not (Test-Path "baseline_comparison\stgcn\venv")) {
    python -m venv baseline_comparison\stgcn\venv
    & baseline_comparison\stgcn\venv\Scripts\Activate.ps1
    pip install -r baseline_comparison\stgcn\requirements.txt
    deactivate
    Write-Host "ST-GCN environment created!" -ForegroundColor Green
}
else {
    Write-Host "ST-GCN environment already exists." -ForegroundColor Gray
}

Write-Host "`nAll advanced baseline environments ready!" -ForegroundColor Cyan
Write-Host "`nTo train models:" -ForegroundColor White
Write-Host "  Transformer: baseline_comparison\transformer\venv\Scripts\Activate.ps1; python baseline_comparison\transformer\train_transformer.py --viewpoint front" -ForegroundColor Gray
Write-Host "  EdgeConv:    baseline_comparison\edgeconv\venv\Scripts\Activate.ps1; python baseline_comparison\edgeconv\train_edgeconv.py --viewpoint front" -ForegroundColor Gray
Write-Host "  ST-GCN:      baseline_comparison\stgcn\venv\Scripts\Activate.ps1; python baseline_comparison\stgcn\train_stgcn.py --viewpoint front" -ForegroundColor Gray
