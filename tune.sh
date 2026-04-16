export HIP_VISIBLE_DEVICES=7

python3 setup.py develop
python3 test_fp8_paged_mqa_logits.py --tune