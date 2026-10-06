#!/bin/sh
# Record GPU and CPU temperatures every 30 s (read-only) into ~/dimension42-explorer/temps.csv
# until ~/dimension42-explorer/temps.stop exists or 10 hours have passed.
cd ~/dimension42-explorer || exit 1
end=$(( $(date +%s) + 36000 ))
while [ ! -e temps.stop ] && [ "$(date +%s)" -lt "$end" ]; do
  t=$(date +%H:%M:%S)
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,fan.speed,clocks_throttle_reasons.active \
      --format=csv,noheader,nounits | while IFS=, read i n tc p f thr; do
        echo "$t,gpu$i,$(echo $n | tr -d ' '),core,$tc,power,$p,fan,$f,throttle,$thr"; done
  fi
  for d in /sys/class/drm/card*/device; do
    hw=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$hw" ] && [ -e $hw/temp2_input ] || continue
    echo "$t,$(basename $(dirname $d)),amd,edge,$(($(cat $hw/temp1_input)/1000)),junction,$(($(cat $hw/temp2_input)/1000)),mem,$(($(cat $hw/temp3_input 2>/dev/null || echo 0)/1000)),power,$(($(cat $hw/power1_average 2>/dev/null || cat $hw/power1_input 2>/dev/null || echo 0)/1000000))"
  done
  for z in /sys/class/hwmon/hwmon*; do
    case $(cat $z/name) in coretemp|k10temp)
      max=0; for x in $z/temp*_input; do v=$(($(cat $x)/1000)); [ $v -gt $max ] && max=$v; done
      echo "$t,cpu,$(cat $z/name),max,$max";;
    esac
  done
  sleep 30
done >> temps.csv
