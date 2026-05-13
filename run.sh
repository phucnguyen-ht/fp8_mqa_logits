export HIP_VISIBLE_DEVICES=7

python3 setup.py develop

# python3 test_fp8_paged_mqa_logits.py --batch 1-8 --use_tuned -kv_length 65536 -mtp 0 --tuned_csv oldtune/paged_fp8_mqa_logits_tuned.csv 
python3 test_fp8_paged_mqa_logits.py --batch 8 -kv_length 65536 -mtp 0 --tuned_csv oldtune/paged_fp8_mqa_logits_tuned.csv