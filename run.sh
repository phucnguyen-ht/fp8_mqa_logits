export HIP_VISIBLE_DEVICES=7

python3 setup.py develop

# for BATCH in $(seq 1 20); do
python3 test_fp8_paged_mqa_logits.py --batch 8 -kv_length 70000 -mtp 0
# done