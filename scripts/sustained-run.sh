#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <video> <model>" >&2
    exit 2
fi

video=$1
model=$2
warmup_seconds=120
interval_seconds=60
sample_count=11
# Samples run from the warm-up mark to warm-up + 10 intervals. The process is
# given a margin beyond the final sample so the last sample cannot race exit.
final_sample_at=$((warmup_seconds + (sample_count - 1) * interval_seconds))
duration_seconds=$((final_sample_at + 30))
# Samples land on a wall-clock second, so allow one second of drift either way
# rather than demanding an exact 60.
interval_tolerance=2
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

# The binary suppresses colour when stdout is not a terminal, but strip any
# escape sequences anyway so parsing never depends on that.
strip_ansi() {
    sed -e 's/\x1b\[[0-9;]*m//g'
}

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

    rss_kb=$(awk '/^VmRSS:/ { print $2; exit }' "/proc/$pid/status" || true)
    progress=$(strip_ansi < "$log_file" \
        | grep 'tracking progress' \
        | tail -n 1 \
        | sed -n 's/.*frame_count=\([0-9][0-9]*\).*live_tracks=\([0-9][0-9]*\).*confirmed_tracks=\([0-9][0-9]*\).*/\1,\2,\3/p' || true)
    if [[ -z "$rss_kb" || -z "$progress" ]]; then
        echo "missing RSS or progress metrics at ${target_elapsed}s; logs: $log_file" >&2
        exit 1
    fi

    actual_elapsed=$(( $(date +%s) - started_at ))
    printf '%s,%s,%s\n' "$actual_elapsed" "$rss_kb" "$progress" >> "$csv_file"
done

cleanup
pid=
trap - EXIT

awk -F, -v expected="$sample_count" -v interval="$interval_seconds" -v tol="$interval_tolerance" '
    NR == 1 { next }
    {
        elapsed[++count] = $1
        rss[count] = $2
        live[count] = $4
    }
    END {
        status = 0
        if (count != expected) {
            print "FAIL: expected " expected " samples, found " count
            exit 1
        }
        for (i = 2; i <= count; i++) {
            gap = elapsed[i] - elapsed[i - 1]
            if (gap < interval - tol || gap > interval + tol) {
                print "FAIL: samples " (i - 1) " and " i " are " gap "s apart (expected " interval "s +/- " tol ")"
                status = 1
            }
        }

        allowance = rss[1] * 0.05
        if (allowance < 10240) allowance = 10240
        if (rss[count] > rss[1] + allowance) {
            printf "FAIL: final RSS %d kB exceeds post-warm-up %d kB + allowance %d kB\n", rss[count], rss[1], allowance
            status = 1
        }

        rss_climbing = 1
        start = count - 3; if (start < 2) start = 2
        for (i = start; i <= count; i++) {
            if (rss[i] <= rss[i - 1]) rss_climbing = 0
        }
        if (rss_climbing) {
            print "FAIL: final five RSS samples are strictly increasing"
            status = 1
        }

        # Live track count is an independent failure condition: a count that
        # climbs with frame count fails the run whatever RSS did.
        live_min = live[1]; live_max = live[1]
        for (i = 1; i <= count; i++) {
            if (live[i] < live_min) live_min = live[i]
            if (live[i] > live_max) live_max = live[i]
        }
        live_climbing = 1
        for (i = start; i <= count; i++) {
            if (live[i] <= live[i - 1]) live_climbing = 0
        }
        if (live_climbing) {
            print "FAIL: final five live track counts are strictly increasing"
            status = 1
        }
        printf "live track count: min=%d max=%d first=%d final=%d\n", live_min, live_max, live[1], live[count]
        printf "RSS: post-warm-up=%d kB final=%d kB delta=%+d kB\n", rss[1], rss[count], rss[count] - rss[1]

        if (status == 0) print "PASS: RSS and live track count criteria satisfied."
        exit status
    }
' "$csv_file"

echo "CSV: $csv_file"
echo "Log: $log_file"
