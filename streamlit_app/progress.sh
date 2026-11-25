#!/bin/zsh
# Live progress for the reach experiment.
#   watch it:   ./streamlit_app/progress.sh -w
#   one shot:   ./streamlit_app/progress.sh
# Sage's kernel solve gives no callback, so the within-cell bar is an ESTIMATE from
# elapsed time against a cost model calibrated on the measured d=4 run (275s for a
# 3050x2970 solve, scaling ~cubically in matrix dimension).
DATA=/Users/gamma/CodeBase/Histogram/streamlit_app/data/reach.jsonl
LOG=$(ls -t /private/tmp/claude-501/*/*/tasks/*.output 2>/dev/null | xargs grep -l "=== d=" 2>/dev/null | head -1)

bar() {  # pct width
  local pct=$1 w=${2:-34} filled
  filled=$(( pct * w / 100 )); (( filled > w )) && filled=$w; (( filled < 0 )) && filled=0
  printf '['
  for i in {1..$w}; do (( i <= filled )) && printf '#' || printf '.'; done
  printf '] %3d%%' $pct
}

render() {
  clear 2>/dev/null
  echo "reach experiment -- how far past u_max=13 does each d go?"
  echo
  local done_cells=0
  if [[ -f $DATA ]]; then
    done_cells=$(wc -l < $DATA | tr -d ' ')
    echo "completed cells:"
    python3 - <<'PY'
import json,os
p="/Users/gamma/CodeBase/Histogram/streamlit_app/data/reach.jsonl"
if os.path.exists(p):
    for l in open(p):
        try: r=json.loads(l)
        except: continue
        mark = "RECOVERED" if r.get("exact") else "failed"
        print(f"   d={r['d']} t={r['t']:>2}  kernel {r['kernel']:>4}  "
              f"genuine {r['genuine_recovered']}/{r['t']}  {mark}  "
              f"({r['sec_total']:.0f}s)")
PY
  else
    echo "   (none yet)"
  fi
  echo
  local pid=$(pgrep -f reach.py | head -1)
  if [[ -z $pid ]]; then
    echo "run finished (or not started)"
    return
  fi
  # macOS ps has no etimes; parse etime ([[dd-]hh:]mm:ss) into seconds
  local raw=$(ps -o etime= -p $pid | tr -d ' ')
  local et=$(python3 -c "
import sys
s=sys.argv[1]; d=0
if '-' in s: d,s=s.split('-'); d=int(d)
p=[int(x) for x in s.split(':')]
while len(p)<3: p.insert(0,0)
print(d*86400 + p[0]*3600 + p[1]*60 + p[2])" "$raw" 2>/dev/null || echo 0)
  local cur=$(grep -E "^=== d=" $LOG 2>/dev/null | tail -1)
  # seconds already spent on finished cells
  local spent=0
  [[ -f $DATA ]] && spent=$(python3 -c "
import json;print(int(sum(json.loads(l)['sec_total'] for l in open('$DATA'))))" 2>/dev/null || echo 0)
  local cell_el=$(( et - spent ))
  local EST=950        # d=5 cell estimate, seconds (calibrated from d=4's 275s)
  local pct=$(( cell_el * 100 / EST ))
  (( pct > 99 )) && pct=99
  echo "current: ${cur:-starting}"
  printf '  cell  '; bar $pct; printf '   %dm%02ds of ~%dm est\n' $(( cell_el/60 )) $(( cell_el%60 )) $(( EST/60 ))
  printf '  total elapsed %dm%02ds\n' $(( et/60 )) $(( et%60 ))
  echo
  echo "(within-cell % is an estimate; Sage's solver reports no progress)"
}

if [[ "$1" == "-w" ]]; then
  while true; do render; sleep 10; done
else
  render
fi
