"""Peak temperatures per component from the temps.csv logs on all nodes (written by temp_log.sh)."""
import subprocess

LIMITS = {"nvidia": ("core", 90, "NVIDIA slows down at ~90-95 C"), "amd": ("junction", 110, "AMD critical at 110 C")}
for node in ("adler40", "knecht24", "specht32", "falke64"):
    rows = subprocess.run(["ssh", "-o", "BatchMode=yes", node, "cat ~/dimension42-explorer/temps.csv"],
                          capture_output=True, text=True).stdout.splitlines()
    peak, throttles, first, last = {}, {}, None, None
    for r in rows:
        f = [x.strip() for x in r.split(",")]
        if len(f) < 5:
            continue
        t, dev = f[0], f[1]
        first, last = first or t, t
        vals = dict(zip(f[3::2], f[4::2]))
        if dev.startswith("gpu"):
            key, temp = f"{dev} {f[2].replace('NVIDIAGeForce', '')}", int(vals["core"])
            if vals.get("throttle", "0x0") not in ("0x0000000000000000", "0x0"):
                throttles[key] = throttles.get(key, 0) + 1
        elif f[2] == "amd":
            key, temp = f"{dev} AMD (hotspot)", int(vals["junction"])
        else:
            key, temp = f"CPU {f[2]}", int(vals["max"])
        if temp > peak.get(key, (0, ""))[0]:
            peak[key] = (temp, t)
    print(f"{node}: {len(rows)} samples {first}-{last}")
    for k, (v, t) in sorted(peak.items()):
        warn = "  <-- WARNING" if ("AMD" in k and v >= 105) or (k.startswith("gpu") and v >= 88) else ""
        print(f"   {k:28s} max {v:3d} C at {t}" + (f", throttled in {throttles[k]} samples" if k in throttles else "") + warn)
