# Dimension42 - start the virtual 4-PC cluster (SPECHT, ADLER, KNECHT, FALKE) in QEMU.
# Each PC opens in its own window. Click a window and use the arrow keys to explore.
# Close the windows (or press Ctrl+C here) to stop.

$qemu = "C:\Program Files\qemu\qemu-system-x86_64.exe"
$root = $PSScriptRoot
$img  = Join-Path $root "build\dimension42.img"

python (Join-Path $root "tools\build.py")
if ($LASTEXITCODE -ne 0) { exit 1 }

# virtual ethernet hub that connects the four PCs
$hub = Start-Process python -ArgumentList "`"$(Join-Path $root 'tools\switch.py')`"" -PassThru -WindowStyle Hidden

$names = @{ 1 = "SPECHT"; 2 = "ADLER"; 3 = "KNECHT"; 4 = "FALKE" }
$vms = foreach ($n in 1..4) {
    Start-Process $qemu -PassThru -ArgumentList @(
        "-name", $names[$n], "-m", "32", "-snapshot",
        "-drive", "file=`"$img`",format=raw",
        "-netdev", "dgram,id=net0,local.type=inet,local.host=127.0.0.1,local.port=$(4200 + $n),remote.type=inet,remote.host=127.0.0.1,remote.port=4200",
        "-device", "rtl8139,netdev=net0,mac=52:54:00:d4:20:0$n"
    )
}

Write-Host "Cluster running. Close all four QEMU windows to stop."
try { $vms | Wait-Process } finally { Stop-Process -Id $hub.Id -ErrorAction SilentlyContinue }
