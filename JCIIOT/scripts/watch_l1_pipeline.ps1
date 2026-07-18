# Usage (from Windows host, repo root or JCIIOT):
#   powershell -File JCIIOT\scripts\watch_l1_pipeline.ps1
#   powershell -File JCIIOT\scripts\watch_l1_pipeline.ps1 -Lines 120
param(
    [int]$Lines = 80,
    [string]$Container = "cvpr-dev"
)

Write-Host "Following L1 pipeline log in $Container (Ctrl+C to stop)..."
docker exec -it $Container bash /workspace/collj/JCIIOT/scripts/tail_l1_pipeline_log.sh $Lines
