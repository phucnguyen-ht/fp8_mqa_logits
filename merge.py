#!/usr/bin/env python3
"""
Combine per-pmc rocprofv2 CSV outputs into a single CSV per scenario.

For pmc_1 (containing FetchSize and WriteSize), also computes:
  - Duration(us)      = (End_Timestamp - Start_Timestamp) / 1000
  - MemoryBandwidth(GB/s) = (FetchSize + WriteSize) * 1.024 / Duration(us)
"""

import argparse
import glob
import os
import re

import pandas as pd

COMMON_COLS = [
    "Dispatch_ID", "GPU_ID", "Queue_ID", "PID", "TID",
    "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup", "Scratch_Per_Workitem",
    "Arch_VGPR", "Accum_VGPR", "SGPR", "Wave_Size", "Kernel_Name",
    "Start_Timestamp", "End_Timestamp", "Correlation_ID",
]


def parse_counters(counters_file):
    """Return list of counter-name lists, one per pmc line."""
    pmcs = []
    with open(counters_file) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("pmc:"):
                continue
            counters = [c.strip() for c in line[4:].split(",") if c.strip()]
            pmcs.append(counters)
    return pmcs


def find_csv(directory):
    """Return the single CSV file inside directory (there should be exactly one)."""
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {directory}")
    return files[0]


def load_pmc1(csv_path):
    """Load pmc_1 CSV and add Duration(us) and MemoryBandwidth(GB/s) columns."""
    df = pd.read_csv(csv_path)
    df["Duration(us)"] = (df["End_Timestamp"].astype(int) - df["Start_Timestamp"].astype(int)) / 1000.0
    df["MemoryBandwidth(GB/s)"] = (
        (df["FetchSize"] + df["WriteSize"]) * 1.024 / df["Duration(us)"]
    )
    return df


def combine_scenario(scenario_dir, pmcs):
    """
    Merge all pmc CSVs for one scenario into a single DataFrame.
    Rows are matched by Dispatch_ID order.
    """
    pmc_dfs = []
    for i, counters in enumerate(pmcs, start=1):
        pmc_dir = os.path.join(scenario_dir, f"pmc_{i}")
        csv_path = find_csv(pmc_dir)
        if i == 1:
            df = load_pmc1(csv_path)
        else:
            df = pd.read_csv(csv_path)
        pmc_dfs.append((i, counters, df))

    # Base: common columns + pmc_1 extras from first dataframe
    base_df = pmc_dfs[0][2]
    extra_pmc1 = ["Duration(us)", "MemoryBandwidth(GB/s)"] + pmc_dfs[0][1]
    keep_cols = [c for c in COMMON_COLS if c in base_df.columns] + [
        c for c in extra_pmc1 if c in base_df.columns
    ]
    result = base_df[keep_cols].copy()

    # Join counter columns from pmc_2 onwards
    for i, counters, df in pmc_dfs[1:]:
        counter_cols = [c for c in counters if c in df.columns]
        result = result.merge(
            df[["Dispatch_ID"] + counter_cols],
            on="Dispatch_ID",
            how="left",
            suffixes=("", f"_pmc{i}"),
        )

    return result


ANALYSIS_META_COLS = [
    "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup", "Scratch_Per_Workitem",
    "Arch_VGPR", "Accum_VGPR", "SGPR", "Wave_Size", "Kernel_Name",
]


def make_analyzed(df, pmcs):
    """Average metric columns per (Scenario, Kernel_Name), keeping meta cols."""
    all_counters = [c for pmc in pmcs for c in pmc]
    metric_cols = ["Duration(us)", "MemoryBandwidth(GB/s)"] + all_counters
    metric_cols = [c for c in metric_cols if c in df.columns]

    # Meta cols are identical for the same kernel — just take first occurrence
    meta_no_kernel = [c for c in ANALYSIS_META_COLS if c != "Kernel_Name"]
    meta = df[["Scenario", "Kernel_Name"] + meta_no_kernel].drop_duplicates(
        subset=["Scenario", "Kernel_Name"]
    )
    avg = df.groupby(["Scenario", "Kernel_Name"], sort=False)[metric_cols].mean().reset_index()
    result = meta.merge(avg, on=["Scenario", "Kernel_Name"], how="right")
    # Ensure column order: Scenario, Kernel_Name, remaining meta, metrics
    ordered_cols = ["Scenario", "Kernel_Name"] + meta_no_kernel + metric_cols
    return result[[c for c in ordered_cols if c in result.columns]]


def main():
    parser = argparse.ArgumentParser(
        description="Combine rocprofv2 per-pmc CSV outputs into one CSV."
    )
    parser.add_argument(
        "--profile-dir", default="mla_profile",
        help="Root profile directory (default: mla_profile)",
    )
    parser.add_argument(
        "--counters", default="counters.txt",
        help="Path to counters.txt (default: counters.txt)",
    )
    parser.add_argument(
        "--output-dir", default="combined_results",
        help="Output directory for per-scenario CSVs (default: combined_results)",
    )
    parser.add_argument(
        "--kernel-filter", nargs="*", default=["mla_a8w8_qh16_qseqlen1_gqaratio16", "_fwd_kernel_stage2_asm"],
        metavar="PATTERN",
        help="Only keep rows whose Kernel_Name contains any of these substrings. "
             "Pass no value to disable filtering. "
             "(default: mla_a8w8_qh16_qseqlen1_gqaratio16 _fwd_kernel_stage2_asm)",
    )
    args = parser.parse_args()

    pmcs = parse_counters(args.counters)
    print(f"Found {len(pmcs)} pmc groups: {pmcs}")

    # scenario_dirs = sorted(
    #     (d for d in glob.glob(os.path.join(args.profile_dir, "*"))
    #      if os.path.isdir(d) and os.path.isdir(os.path.join(d, "pmc_1"))),
    #     key=lambda d: [int(x) for x in re.split(r"[:\-_]", os.path.basename(d)) if x.isdigit()],
    # )
    # if not scenario_dirs:
    #     raise RuntimeError(f"No scenario directories found in {args.profile_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    analyzed_frames = []

    # for scenario_dir in scenario_dirs:
    scenario_dir = args.profile_dir
    scenario = os.path.basename(scenario_dir)
    print(f"Processing scenario: {scenario}")
    df = combine_scenario(scenario_dir, pmcs)
    df.insert(0, "Scenario", scenario)

    if args.kernel_filter:
        pattern = "|".join(re.escape(p) for p in args.kernel_filter)
        df = df[df["Kernel_Name"].str.contains(pattern, na=False)]
        print(f"  (filtered to {len(df)} rows matching kernel patterns)")

    # Save per-scenario CSV (replace path separators so filename is safe)
    safe_name = re.sub(r"[/\\]", "_", scenario)
    scenario_csv = os.path.join(args.output_dir, f"{safe_name}.csv")
    df.to_csv(scenario_csv, index=False)
    print(f"  -> {scenario_csv} ({len(df)} rows)")

    analyzed_frames.append(make_analyzed(df, pmcs))

    analyzed = pd.concat(analyzed_frames, ignore_index=True)
    analyzed_csv = os.path.join(args.output_dir, "analyzed.csv")
    analyzed.to_csv(analyzed_csv, index=False)
    print(f"Analyzed: {analyzed_csv} ({len(analyzed)} rows)")


if __name__ == "__main__":
    main()