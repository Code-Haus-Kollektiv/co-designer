# Check if the CUDA_PATH environment variable is defined.
if ($env:CUDA_PATH) {
    Write-Host "CUDA_PATH environment variable is defined: $($env:CUDA_PATH)"
    
    # Check if the nvcc compiler exists in the expected directory.
    $nvccPath = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (Test-Path $nvccPath) {
        Write-Host "Found nvcc at: $nvccPath"
    }
    else {
        Write-Host "nvcc.exe not found in CUDA_PATH/bin. This may indicate an incomplete CUDA installation." -ForegroundColor Yellow
    }
}
else {
    Write-Host "CUDA_PATH environment variable is not set. CUDA may not be installed." -ForegroundColor Red
}

# Check if nvidia-smi is available (common indicator of NVIDIA driver & CUDA runtime being installed).
try {
    $nvidiaSmiOutput = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "nvidia-smi command executed successfully. GPU(s) detected:"
        Write-Host $nvidiaSmiOutput
    }
    else {
        Write-Host "nvidia-smi command did not execute as expected. CUDA might not be properly installed or configured in the PATH." -ForegroundColor Yellow
    }
}
catch {
    Write-Host "Error running nvidia-smi. This command may not be in your PATH or CUDA may not be installed." -ForegroundColor Red
}