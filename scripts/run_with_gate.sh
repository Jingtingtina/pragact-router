
set -euo pipefail


: "${PRAGACT_FAST_CUES:=1}"
: "${PRAGACT_FAST_TAU:=0.30}"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <config_yaml> <data_jsonl>" >&2
  exit 2
fi

echo "FAST_CUES=$PRAGACT_FAST_CUES  TAU=$PRAGACT_FAST_TAU"
python -m scripts.eval_confusion --config "$1" --data "$2"
