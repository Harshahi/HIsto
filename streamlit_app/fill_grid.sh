#!/bin/zsh
# Fill the (d, t, sigma) grid in priority order, boundary first.
#
#     DEADLINE_HR=7 ./streamlit_app/fill_grid.sh
#
# Tiers are ordered so that if the clock runs out, what is already on disk is the
# part worth talking about: the behaviour at t = 13..15, where the bundling cliff
# is, rather than another decimal place at t = 4 where everything works.
#
# cliff_probe.py appends each decode to data/cliff_3_3_3.jsonl as it finishes and
# skips seeds already there, so this is safe to interrupt and safe to re-run.
set -u
cd "$(dirname "$0")/.."

DEADLINE_HR=${DEADLINE_HR:-7}
WORKERS=${WORKERS:-8}
START=$(date +%s)
LOG=${LOG:-/tmp/fill_grid.log}

ALL_T="1,2,3,4,5,6,7,8,9,10,11,12,13"
SB="0,1.2,1.4"            # boundary sigmas
SF="0,1.0,1.2,1.4,1.6"    # full sweep sigmas

tier () {   # name  D_LIST  T_LIST  SIGMAS  TRIALS  BUDGET
  local name=$1 d=$2 ts=$3 sg=$4 n=$5 bud=$6
  local now=$(( $(date +%s) - START ))
  local left=$(( DEADLINE_HR * 3600 - now ))
  if (( left < 300 )); then
    echo "[$(date +%H:%M:%S)] SKIP $name -- only ${left}s left of budget" >> "$LOG"
    return
  fi
  echo "[$(date +%H:%M:%S)] >>> $name  (d=$d t=$ts sig=$sg x$n, ${left}s left)" >> "$LOG"
  D_LIST=$d T_LIST=$ts SIGMAS=$sg TRIALS=$n WORKERS=$WORKERS BUDGET=$bud \
    MAX_BY_D= TRIALS_BY_D= \
    uv run --group sage --group app python streamlit_app/cliff_probe.py >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] <<< $name done" >> "$LOG"
}

# --- the boundary, cheapest first -----------------------------------------
tier "boundary d=2,3 t=14,15"  2,3  14,15  "$SB"  5  3600
# a full cheap sweep early, so the table reads as populated rather than sparse
tier "bulk d=3 t=1..13"        3    "$ALL_T" "$SF" 10 1800
# the open question: d=4 past u_max, under noise. gs_bound says the kernel is
# 27-dimensional at t=14, sigma=0 -- nobody has run it with noise on.
tier "boundary d=4 t=14,15"    4    14,15  "$SB"  3  7200
tier "bulk d=4 t=1..13"        4    "$ALL_T" "$SF" 5  2400
tier "boundary d=5 t=14,15"    5    14,15  "$SB"  3  10800
tier "bulk d=5 t=1..13"        5    "$ALL_T" "$SF" 3  3600
tier "boundary d=6 t=14,15"    6    14,15  "0,1.2" 2 14400
tier "bulk d=6 t=1..13"        6    "$ALL_T" "$SF" 2  4800

echo "[$(date +%H:%M:%S)] ALL TIERS DONE after $(( ($(date +%s) - START) / 60 )) min" >> "$LOG"
