<#
.SYNOPSIS
    VoxMorph RVC 模型下载脚本

.DESCRIPTION
    下载 RVC 变声所需的两个 ONNX 模型：
      1. ContentVec (content_vec_500.onnx) — 内容特征提取
      2. RMVPE (rmvpe.onnx) — F0 基频估计

    RVC 变声模型 (.onnx) 需用户自行准备。

.PARAMETER OutputDir
    模型输出目录（默认: models/）

.PARAMETER Force
    强制重新下载（即使文件已存在）

.EXAMPLE
    pwsh .\scripts\download-models.ps1
    pwsh .\scripts\download-models.ps1 -OutputDir D:\models -Force
#>

param(
    [string]$OutputDir = "models",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# 模型下载 URL（第三方分发，GPL-3.0 许可）。
$ModelUrls = @{
    "content_vec_500.onnx" = "https://huggingface.co/therealvinter/ContentVec/resolve/main/content_vec_500.onnx"
    "rmvpe.onnx"           = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx"
}

Write-Host "=" * 60
Write-Host "VoxMorph RVC 模型下载"
Write-Host "=" * 60
Write-Host ""

# 创建输出目录。
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$allOk = $true

foreach ($entry in $ModelUrls.GetEnumerator()) {
    $filename = $entry.Key
    $url = $entry.Value
    $outputPath = Join-Path $OutputDir $filename

    Write-Host "[$filename]"
    Write-Host "  URL: $url"
    Write-Host "  目标: $outputPath"

    if ((Test-Path $outputPath) -and -not $Force) {
        Write-Host "  已存在，跳过（使用 -Force 强制重新下载）"
        Write-Host ""
        continue
    }

    try {
        # 下载文件。
        $ProgressPreference = 'Continue'
        Invoke-WebRequest -Uri $url -OutFile $outputPath -UseBasicParsing
        $fileInfo = Get-Item $outputPath
        $sizeMB = [math]::Round($fileInfo.Length / 1MB, 1)
        Write-Host "  下载完成: ${sizeMB}MB"
    }
    catch {
        Write-Host "  下载失败: $_"
        $allOk = $false
    }

    Write-Host ""
}

# RVC 模型提示。
Write-Host "=" * 60
Write-Host "RVC 变声模型 (.onnx)"
Write-Host "=" * 60
Write-Host ""
Write-Host "RVC 变声模型需要您自行准备："
Write-Host "  1. 从 RVC 项目或 VCClient 等工具导出 .onnx 格式的 RVC 模型"
Write-Host "  2. 将 .onnx 文件放入 $OutputDir 目录"
Write-Host "  3. 在 VoxMorph GUI 中选择该模型文件"
Write-Host ""
Write-Host "注意: .pth 格式的 RVC 模型不能直接使用，需先转换为 .onnx。"
Write-Host ""

if ($allOk) {
    Write-Host "模型下载完成！"
    Write-Host "模型保存在: $(Resolve-Path $OutputDir)"
    exit 0
} else {
    Write-Host "部分模型下载失败，请检查网络连接后重试。"
    exit 1
}
