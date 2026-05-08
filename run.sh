export HIP_VISIBLE_DEVICES=1

python3 setup.py develop

python3 test_fp8_paged_mqa_logits.py --batch 1-8 --use_tuned -kv_length 512,1024,2048,4096,8192,16384,32768,65536 -mtp 0 --tuned_csv oldtune/paged_fp8_mqa_logits_tuned.csv 