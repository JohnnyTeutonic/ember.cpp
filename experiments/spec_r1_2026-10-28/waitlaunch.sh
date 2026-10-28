#!/bin/bash
# Waits for a free Colab GPU slot (fewer than three sessions in the listing, no [?] orphan), then runs drive.sh once
# (run in WSL, foreground):   bash waitlaunch.sh <session> <max-minutes> <stage dir> <driver log> [max hours to wait]
# The listing is read every WAIT_S seconds (default 300); drive.sh asks for an L4 and stops if none is granted, after which this tries
# again until the wait limit.
S=${1:?session}; MAXMIN=${2:?max minutes}; ST=${3:?stage}; LOG=${4:?log}; HOURS=${5:-8}
X=$(dirname "$(readlink -f "$0")")
end=$(( $(date +%s) + HOURS * 3600 ))
while [ $(date +%s) -lt $end ]; do
  l=$(timeout 180 ~/.local/bin/colab sessions 2>&1)
  n=$(echo "$l" | grep -c "Hardware:")
  if ! echo "$l" | grep -q "\[?\]" && [ "$n" -lt 3 ]; then
    echo "$(date +%H:%M:%S) $n sessions listed: launching" >> "$LOG.wait"
    bash $X/drive.sh "$S" "$MAXMIN" "$ST" > "$LOG" 2>&1
    grep -q "no L4" "$LOG" || exit 0
    echo "$(date +%H:%M:%S) no L4 granted; waiting again" >> "$LOG.wait"
  else
    echo "$(date +%H:%M:%S) $n sessions listed: waiting" >> "$LOG.wait"
  fi
  sleep ${WAIT_S:-300}
done
echo "$(date +%H:%M:%S) gave up after $HOURS h" >> "$LOG.wait"
exit 1
