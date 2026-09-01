#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <video> <model>" >&2
    exit 2
fi

video=$1
model=$2
duration_seconds=720
warmup_seconds=120
interval_seconds=60
sample_count=11
release_bin=target/release/vision-engine
output_dir=$(mktemp -d "${TMPDIR:-/tmp}/vision-engine-ve011.XXXXXX")
log_file="$output_dir/runtime.log"
csv_file="$output_dir/sustained-run.csv"

cleanup() {
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ ! -x "$release_bin" ]]; then
    echo "release binary not found: $release_bin (run cargo build --release first)" >&2
    exit 2
fi

printf 'elapsed_seconds,rss_kb,frame_count,live_track_count,confirmed_track_count\n' > "$csv_file"
"$release_bin" "$video" --model "$model" --loop-for-seconds "$duration_seconds" >"$log_file" 2>&1 &
pid=$!
started_at=$(date +%s)

for sample in $(seq 0 $((sample_count - 1))); do
    target_elapsed=$((warmup_seconds + sample * interval_seconds))
    now_elapsed=$(( $(date +%s) - started_at ))
    if (( now_elapsed < target_elapsed )); then
        sleep "$((target_elapsed - now_elapsed))"
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "runtime exited before sample $((sample + 1)) at ${target_elapsed}s; logs: $log_file" >&2
        exit 1
    fi

    rss_kb=$(awk '/^VmRSS:/ { print $2; exit }' "/proc/$pid/status")
    progress=$(grep 'tracking progress' "$log_file" | tail -n 1 | sed -n 's/.*frame_count=\([0-9][0-9]*\).*live_tracks=\([0-9][0-9]*\).*confirmed_tracks=\([0-9][0-9]*\).*/\1,\2,\3/p')
    if [[ -z "$rss_kb" || -z "$progress" ]]; then
        echo "missing RSS or progress metrics at ${target_elapsed}s; logs: $log_file" >&2
        exit 1
    fi

    actual_elapsed=$(( $(date +%s) - started_at ))
    printf '%s,%s,%s\n' "$actual_elapsed" "$rss_kb" "$progress" >> "$csv_file"
done

wait "$pid"
pid=
trap - EXIT

awk -F, '
    NR == 1 { next }
    {
        elapsed[++count] = $1
        rss[count] = $2
    }
    END {
        if (count != 11) {
            print "FAIL: expected 11 samples, found " count
            exit 1
        }
        for (i = 2; i <= count; i++) {
            if (elapsed[i] - elapsed[i - 1] != 60) {
                print "FAIL: samples " (i - 1) " and " i " are not 60 seconds apart"
                exit 1
            }
        }
        allowance = rss[1] * 0.05
        if (allowance < 10240) allowance = 10240
        if (rss[count] > rss[1] + allowance) {
            print "FAIL: final RSS exceeds post-warm-up allowance"
            exit 1
        }
        strictly_increasing = 1
        for (i = count - 4; i <= count; i++) {
            if (i > count - 4 && rss[i] <= rss[i - 1]) strictly_increasing = 0
        }
        if (strictly_increasing) {
            print "FAIL: final five RSS samples are strictly increasing"
            exit 1
        }
        print "PASS: RSS sampling criteria satisfied; inspect live track counts against known empty-scene intervals."
    }
' "$csv_file"

echo "CSV: $csv_file"
echo "Log: $log_file"
