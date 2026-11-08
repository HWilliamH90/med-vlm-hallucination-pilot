# Watchdog that drives the ModelScope download with a hard per-attempt timeout.
# The network on this machine silently stalls long-lived connections, so each
# attempt is killed after 150 s and restarted; ModelScope resumes from the
# existing partial files, so progress accumulates until the model is complete.
$ErrorActionPreference = "Continue"

$wd = "C:\Users\asus\Doubao\chats\2026-11-04\new-chat\med_vlm_work"
$python = "D:\Anaconda\envs\medvlm\python.exe"
$dlScript = Join-Path $wd "scripts\00b_download_model.py"
$watchLog = Join-Path $wd "outputs\watchdog.log"
$attemptOut = Join-Path $wd "outputs\dl_attempt.out"
$attemptErr = Join-Path $wd "outputs\dl_attempt.err"
$cache = "C:\Users\asus\.cache\modelscope"

"Watchdog started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $watchLog -Encoding utf8

for ($attempt = 1; $attempt -le 60; $attempt++) {
    $size = (Get-ChildItem -LiteralPath $cache -Recurse -Force -File -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum
    $line = "[{0}] attempt {1}, size {2} MB" -f (Get-Date -Format "HH:mm:ss"), $attempt, [math]::Round($size/1MB, 1)
    $line | Out-File -FilePath $watchLog -Append -Encoding utf8

    Remove-Item -LiteralPath $attemptOut, $attemptErr -Force -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $python -ArgumentList "`"$dlScript`"" `
        -WorkingDirectory $wd -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $attemptOut -RedirectStandardError $attemptErr

    if ($proc.WaitForExit(150000)) {
        $out = Get-Content -LiteralPath $attemptOut -Raw -ErrorAction SilentlyContinue
        if ($out -match "MODEL_PATH:") {
            "DONE $(Get-Date -Format 'HH:mm:ss') -> $out" | Out-File -FilePath $watchLog -Append -Encoding utf8
            break
        }
        "Process exited early without finishing; retrying" | Out-File -FilePath $watchLog -Append -Encoding utf8
    }
    else {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Get-Process -ErrorAction SilentlyContinue |
            Where-Object { $_.Path -eq $python -and $_.Id -ne $PID } |
            Stop-Process -Force -ErrorAction SilentlyContinue
        "Attempt timed out; killed and restarting (resume)" | Out-File -FilePath $watchLog -Append -Encoding utf8
    }
    Start-Sleep -Seconds 3
}
