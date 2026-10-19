<#
    一键启动 HKC 数据平台
    用法：
        .\start.ps1                 # 只允许本机访问（默认，最安全）
        .\start.ps1 -Lan            # 允许局域网访问（手机/别的电脑打开）
        .\start.ps1 -Port 8080      # 换端口
        .\start.ps1 -NoBrowser      # 不自动开浏览器
#>
param(
    [switch]$Lan,
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$host_ = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }
$env:PYTHONPATH = "$PWD\.vendor"
$env:PYTHONIOENCODING = "utf-8"

# 依赖自检
if (-not (Test-Path ".vendor\fastapi")) {
    Write-Host "✗ 找不到 .vendor\fastapi —— 依赖目录缺失或被移动了" -ForegroundColor Red
    exit 1
}

# 端口占用检查
$busy = netstat -ano | Select-String ":$Port\s+.*LISTENING"
if ($busy) {
    Write-Host "⚠ 端口 $Port 已被占用，可能平台已经在跑：$($busy[0].ToString().Trim())" -ForegroundColor Yellow
    Write-Host "  直接打开 http://127.0.0.1:$Port ；要重启请先关掉那个窗口（Ctrl+C）" -ForegroundColor Yellow
    exit 0
}

$shown = if ($Lan) { "http://$((Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
    Select-Object -First 1).IPAddress):$Port" } else { "http://127.0.0.1:$Port" }

Write-Host ""
Write-Host "  HKC 数据平台启动中…" -ForegroundColor Cyan
Write-Host "  地址 : $shown   （本机也可用 http://127.0.0.1:$Port）"
Write-Host "  登录 : admin / admin123"
Write-Host "  停止 : 在本窗口按 Ctrl+C"
Write-Host ""
if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$Port" }

python -m uvicorn main:app --host $host_ --port $Port
