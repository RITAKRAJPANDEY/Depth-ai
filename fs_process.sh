#!/usr/bin/env bash
# Runs FoundationStereo on the newest recording (in WSL) and copies the comparison video + summary to Windows.
# Arguments are passed to fs_process_recording.py, e.g.:  --model vitl --iters 16 --stride 2
set -e
cd "$(dirname "$0")"
rec=$(ls -td ~/oakffc3p/recordings/*/ | head -1)
~/oakffc3p/.venv-fs/bin/python -u fs_process_recording.py "$rec" "$@" 2>&1 \
    | grep --line-buffered -vE "FutureWarning|autocast|USB protocol|warnings.warn"
out="recordings/$(basename "$rec")"
mkdir -p "$out"
cp "$rec"/compare_*.avi "$rec"/fs_summary_*.json "$rec"/meta.json "$out"/
echo "Comparison video and summary copied to $(wslpath -w "$(realpath "$out")")"
echo "Full recording (PNG frames, FoundationStereo depth) is in $(wslpath -w "$rec")"
