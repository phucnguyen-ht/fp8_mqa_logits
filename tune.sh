export HIP_VISIBLE_DEVICES=7

rm -rf *.so *.egg-info
python3 setup.py develop

# Batch sizes 1..32 + 64,128,256 — passed one-at-a-time so each python invocation
# gets a fresh GPU memory state (works around an allocator leak we couldn't pin down).
BATCHES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 64 128 256)

# kv_length: 512, 1024, 2048, 4096, then 8192 + 4096*k for k=0..14 (up to 65536)
KV_LENGTHS=(512 1024 2048 4096 8192 12288 16384 20480 24576 28672 32768 36864 40960 45056 49152 53248 57344 61440 65536)

OUT_DIR="tune_results"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# Single accumulating CSV across all (mtp, kv_length, batch) — Python skips rows already present.
# Placed inside logs/ so all tune outputs (csv + per-run logs) sit under one folder.
CSV_FILE="${LOG_DIR}/tune_results.csv"

# Total wall-clock budget for the whole tune.sh (default 5h30m = 19800s).
# Override via env, e.g. TIMEOUT=7200 ./tune.sh for 2 hours.
# TIMEOUT="${TIMEOUT:-19800}"
TIMEOUT="${TIMEOUT:-60}"
START_TS=$(date +%s)
echo "[budget] TIMEOUT=${TIMEOUT}s ($((TIMEOUT/3600))h$(( (TIMEOUT%3600)/60 ))m), started at $(date -d @${START_TS} '+%F %T')"

stop_reason=""
for mtp in 0 1; do
    for kv_length in "${KV_LENGTHS[@]}"; do
        for batch in "${BATCHES[@]}"; do
            NOW=$(date +%s)
            ELAPSED=$((NOW - START_TS))
            REMAINING=$((TIMEOUT - ELAPSED))
            if [ "${REMAINING}" -le 0 ]; then
                stop_reason="budget exhausted before iteration (elapsed=${ELAPSED}s)"
                break 3
            fi

            RUN_LOG_DIR="${LOG_DIR}/mtp_${mtp}/kvlength_${kv_length}"
            mkdir -p "${RUN_LOG_DIR}"
            RUN_LOG="${RUN_LOG_DIR}/run.log"

            echo "------------------------------------------------"
            echo "Tuning: mtp=${mtp}, kv_length=${kv_length}, batch=${batch}"
            echo "  csv       -> ${CSV_FILE}"
            echo "  log_dir   -> ${RUN_LOG_DIR}"
            echo "  elapsed   -> ${ELAPSED}s / ${TIMEOUT}s   (remaining ${REMAINING}s)"
            echo "------------------------------------------------"

            # Hard-cap this python invocation at REMAINING seconds. SIGTERM first, then
            # SIGKILL 30s later if it doesn't exit. Exit code 124 = timeout was hit.
            timeout --kill-after=30s "${REMAINING}s" \
                python3 test_fp8_paged_mqa_logits.py --tune \
                    --batch "${batch}" \
                    -kv_length "${kv_length}" \
                    -mtp "${mtp}" \
                    --tune_csv "${CSV_FILE}" \
                    --log_dir "${RUN_LOG_DIR}" \
                    >> "${RUN_LOG}" 2>&1
            rc=$?
            if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
                stop_reason="python killed by timeout (rc=$rc) at mtp=${mtp} kv_length=${kv_length} batch=${batch}"
                break 3
            elif [ "$rc" -ne 0 ]; then
                echo "  [warn] python exited with rc=$rc at batch=${batch} (see ${RUN_LOG}); continuing"
            fi
        done
    done
done

TOTAL=$(( $(date +%s) - START_TS ))
echo "================================================"
if [ -n "${stop_reason}" ]; then
    echo "[timeout] Stopped early: ${stop_reason}"
fi
echo "[budget] Total elapsed: ${TOTAL}s / ${TIMEOUT}s"
echo "================================================"
