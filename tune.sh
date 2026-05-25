#!/usr/bin/env bash
# Usage:
#   MODE=tune  VISIBLE_DEVICES=4,5,6,7 [TIMEOUT=10800] ./tune.sh   (default)
#   MODE=bench VISIBLE_DEVICES=4,5,6,7 ./tune.sh

MODE="${MODE:-tune}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPUS <<< "$VISIBLE_DEVICES"
NUM_GPUS=${#GPUS[@]}

# Build once before parallelising (shared .so)
rm -rf *.so *.egg-info build/
python3 setup.py develop

# BATCHES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 64 128 256 512)
VERSION=7

# BATCHES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
# BATCHES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
BATCHES=($(seq 129 256))
# BATCHES=(1 2 3 4 5 6 7 8)
KV_LENGTHS=(65536)
MTP_LIST=(0)

OUT_DIR="tune_results_ver$VERSION"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

TIMEOUT="${TIMEOUT:-2880000}"
START_TS=$(date +%s)

echo "[budget]    TIMEOUT=${TIMEOUT}s ($((TIMEOUT/3600))h$(( (TIMEOUT%3600)/60 ))m), started at $(date -d @${START_TS} '+%F %T')"
echo "[multi-gpu] MODE=${MODE}, ${NUM_GPUS} GPU(s): ${GPUS[*]}"

# Round-robin: assign BATCHES[i] → GPU[i % NUM_GPUS]
declare -A GPU_BATCH_LISTS
for i in "${!GPUS[@]}"; do GPU_BATCH_LISTS[$i]=""; done
for i in "${!BATCHES[@]}"; do
    slot=$((i % NUM_GPUS))
    GPU_BATCH_LISTS[$slot]+="${BATCHES[$i]} "
done

# ============================================================
if [ "${MODE}" = "tune" ]; then
# ============================================================

    FINAL_CSV="${LOG_DIR}/tune_results.csv"
    PIDS=()
    for i in "${!GPUS[@]}"; do
        gpu_id="${GPUS[$i]}"
        batches_str="${GPU_BATCH_LISTS[$i]}"
        [[ -z "${batches_str// /}" ]] && { echo "[gpu ${gpu_id}] no batches assigned, skip"; continue; }

        GPU_CSV="${LOG_DIR}/tune_results_gpu${gpu_id}.csv"
        GPU_LOG_DIR="${LOG_DIR}/gpu${gpu_id}"
        mkdir -p "${GPU_LOG_DIR}"
        echo "[gpu ${gpu_id}] batches: ${batches_str}"

        (
            export HIP_VISIBLE_DEVICES="${gpu_id}"
            for mtp in "${MTP_LIST[@]}"; do
                for kv_length in "${KV_LENGTHS[@]}"; do
                    for batch in ${batches_str}; do
                        REMAINING=$(( TIMEOUT - ($(date +%s) - START_TS) ))
                        [ "${REMAINING}" -le 0 ] && { echo "[gpu ${gpu_id}] budget exhausted"; exit 0; }

                        # Abort this GPU's tune loop if VRAM usage >= 50% (another workload is running).
                        # Retry up to 3 times with 10s pause to tolerate transient spikes.
                        vram_pct=0
                        for attempt in 1 2 3; do
                            vram_pct=$(rocm-smi -d "${gpu_id}" --showmeminfo vram 2>/dev/null \
                                | awk '
                                    /VRAM Total Memory \(B\)/      { total = $NF }
                                    /VRAM Total Used Memory \(B\)/ { used  = $NF }
                                    END { if (total > 0) printf "%.0f", used/total*100; else print 0 }
                                ')
                            if [ "${vram_pct:-0}" -lt 50 ]; then
                                break
                            fi
                            echo "[gpu ${gpu_id}] attempt ${attempt}/3: VRAM ${vram_pct}% used (>= 50%); waiting 10s..."
                            sleep 10
                        done
                        if [ "${vram_pct:-0}" -ge 50 ]; then
                            echo "[gpu ${gpu_id}] VRAM still ${vram_pct}% used after 3 retries; aborting tune on this GPU"
                            exit 0
                        fi

                        RUN_LOG_DIR="${GPU_LOG_DIR}/mtp_${mtp}/kvlength_${kv_length}"
                        mkdir -p "${RUN_LOG_DIR}"

                        timeout --kill-after=30s "${REMAINING}s" \
                            python3 test_fp8_paged_mqa_logits.py --tune \
                                --batch       "${batch}" \
                                -kv_length    "${kv_length}" \
                                -mtp          "${mtp}" \
                                --tune_csv    "${GPU_CSV}" \
                                --resume_csv  "${FINAL_CSV}" \
                                --log_dir     "${RUN_LOG_DIR}" \
                                --version     $VERSION \
                            >> "${GPU_LOG_DIR}/run.log" 2>&1
                        rc=$?
                        [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && { echo "[gpu ${gpu_id}] killed by timeout"; exit 0; }
                        [ "$rc" -ne 0 ] && echo "[gpu ${gpu_id}] warn: rc=$rc at mtp=${mtp} kv=${kv_length} B=${batch}"
                    done
                done
            done
        ) &
        PIDS+=($!)
    done

    echo "[barrier] waiting for ${#PIDS[@]} worker(s)..."
    for pid in "${PIDS[@]}"; do wait "$pid" || true; done
    TOTAL=$(( $(date +%s) - START_TS ))
    echo "[barrier] all done. elapsed: ${TOTAL}s / ${TIMEOUT}s"

    echo "[merge] merging per-GPU CSVs → ${FINAL_CSV}"
    python3 - <<PYEOF
import pandas as pd, glob, os, sys

log_dir   = "${LOG_DIR}"
final_csv = "${FINAL_CSV}"
key_cols  = ["batch_size", "next_n", "heads", "index_dim", "avg_kv_length"]

def merge_pattern(pattern, out_path):
    csvs = sorted(glob.glob(pattern))
    if not csvs:
        print(f"  [merge] no files match {pattern}")
        return
    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
            frames.append(df)
            print(f"  {f}: {len(df)} rows")
        except Exception as e:
            print(f"  {f}: SKIP ({e})")
    if not frames:
        return
    merged = (pd.concat(frames, ignore_index=True)
                .pipe(lambda d: d.assign(**{c: d[c].astype(int) for c in key_cols if c in d}))
                .sort_values(key_cols)
                .drop_duplicates(subset=key_cols, keep="first")
                .reset_index(drop=True))
    merged.to_csv(out_path, index=False)
    print(f"  -> {out_path}: {len(merged)} rows")

merge_pattern(f"{log_dir}/tune_results_gpu*.csv",       final_csv)
merge_pattern(f"{log_dir}/tune_results_gpu*_top1.csv",  os.path.splitext(final_csv)[0] + "_top1.csv")
PYEOF

# ============================================================
elif [ "${MODE}" = "bench" ]; then
# ============================================================

    PIDS=()
    for i in "${!GPUS[@]}"; do
        gpu_id="${GPUS[$i]}"
        batches_str="${GPU_BATCH_LISTS[$i]}"
        [[ -z "${batches_str// /}" ]] && { echo "[gpu ${gpu_id}] no batches assigned, skip"; continue; }

        GPU_LOG_DIR="${LOG_DIR}/bench_gpu${gpu_id}"
        mkdir -p "${GPU_LOG_DIR}"
        echo "[gpu ${gpu_id}] batches: ${batches_str}"

        (
            export HIP_VISIBLE_DEVICES="${gpu_id}"
            batches_arg="${batches_str// /,}"; batches_arg="${batches_arg%,}"
            kv_arg="${KV_LENGTHS[*]}";         kv_arg="${kv_arg// /,}"
            mtp_arg="${MTP_LIST[*]}";           mtp_arg="${mtp_arg// /,}"

            python3 test_fp8_paged_mqa_logits.py \
                --batch    "${batches_arg}" \
                -kv_length "${kv_arg}" \
                -mtp       "${mtp_arg}" \
                --use_tuned \
                --run_csv  "${GPU_LOG_DIR}/run_gpu${gpu_id}.csv" \
                --version  $VERSION \
            > "${GPU_LOG_DIR}/run.log" 2>&1
        ) &
        PIDS+=($!)
    done

    echo "[barrier] waiting for ${#PIDS[@]} worker(s)..."
    for pid in "${PIDS[@]}"; do wait "$pid" || true; done
    TOTAL=$(( $(date +%s) - START_TS ))
    echo "[barrier] all done. elapsed: ${TOTAL}s / ${TIMEOUT}s"

    echo "[merge] merging per-GPU bench CSVs → run.csv"
    python3 - <<PYEOF
import pandas as pd, glob, sys

log_dir  = "${LOG_DIR}"
key_cols = ["batch_size", "next_n", "heads", "index_dim", "avg_kv_length"]

csvs = sorted(glob.glob(f"{log_dir}/bench_gpu*/run_gpu*.csv"))
if not csvs:
    print("  [merge] no bench CSVs found")
    sys.exit(0)
frames = []
for f in csvs:
    try:
        df = pd.read_csv(f)
        frames.append(df)
        print(f"  {f}: {len(df)} rows")
    except Exception as e:
        print(f"  {f}: SKIP ({e})")
if not frames:
    sys.exit(0)
merged = (pd.concat(frames, ignore_index=True)
            .pipe(lambda d: d.assign(**{c: d[c].astype(int) for c in key_cols if c in d}))
            .sort_values(key_cols)
            .reset_index(drop=True))
merged.to_csv("run.csv", index=False)
print(f"  -> run.csv: {len(merged)} rows")
print(merged.to_string(index=False))
PYEOF

else
    echo "[error] unknown MODE=${MODE}. Use tune or bench."
    exit 1
fi
