export HIP_VISIBLE_DEVICES=7

python3 setup.py develop

export bs=8
export profile_dir="profile/v3/conc$bs"
export counters="profile/metrics/combine_set.txt"

# rocprofv3 --att --att-activity 10 -d $profile_dir -- python3 test_fp8_paged_mqa_logits.py --profile --batch $bs --use_tuned -kv_length 65536 -mtp 0 --tuned_csv oldtune/paged_fp8_mqa_logits_tuned.csv
rocprofv3 --att --att-activity 10 -d $profile_dir -- python3 test_fp8_paged_mqa_logits.py --profile --batch $bs -kv_length 65536 -mtp 0 --tuned_csv oldtune/paged_fp8_mqa_logits_tuned.csv

# KERNELS=(
#     "void v2::fp8_paged_mqa_logits_kernel<8, 64>(unsigned char const*, unsigned char const*, float const*, int const*, int const*, float*, int, int, int, int, int, int) (.kd)"
#     "_deepgemm_fp8_paged_mqa_logits.kd"
# )
# python3 merge.py --profile-dir "$profile_dir" --counters "$counters" --output-dir "$profile_dir" \
#     --kernel-filter "${KERNELS[@]}"