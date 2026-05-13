export HIP_VISIBLE_DEVICES=7

python3 setup.py develop

export bs=8
export version=2
export profile_dir="profile/v2/ver$version/conc_1_$bs"
export counters="profile/metrics/combine_set.txt"

rm -rf $profile_dir
mkdir -p $profile_dir

# rocprofv2 -i $counters -d $profile_dir python3 test_fp8_paged_mqa_logits.py --profile --batch $bs --use_tuned -kv_length 65536 -mtp 0 --tuned_csv oldtune/paged_fp8_mqa_logits_tuned.csv --profile
rocprofv2 -i $counters -d $profile_dir python3 test_fp8_paged_mqa_logits.py --profile --batch $bs -kv_length 65536 -mtp 0 --version $version

KERNELS=(
    "void v$version::fp8_paged_mqa_logits_kernel<4, 64>(unsigned char const*, unsigned char const*, float const*, int const*, int const*, float*, int, int, int, int, int, int) (.kd)"
    "_deepgemm_fp8_paged_mqa_logits.kd"
)
python3 merge.py --profile-dir "$profile_dir" --counters "$counters" --output-dir "$profile_dir" \
    --kernel-filter "${KERNELS[@]}"