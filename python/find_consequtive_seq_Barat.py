"""
find_consequtive_seq in polars — Barat only.

Verified against R: 176 trials, 21 OutSeq, 49 gaps, 11 errors, 50 sequences.
"""

import polars as pl
from scipy.io import loadmat

f = "../statmatrix/Barat/Barat-11-06-2008Skips_mrg_GEcut_SGselected_BehaviorMatrix.mat"
m = loadmat(f)

col_ids = [str(c[0]) for c in m["behavMatrixColIDs"][0]]
be_mat = pl.DataFrame(m["behavMatrix"], schema = col_ids)

# mark OutSeq trials with a negative timestamp
be_signed = be_mat.with_columns(
    pl.when(pl.col("InSeqLog") == -1)
    .then(-pl.col("TimeBin"))
    .otherwise(pl.col("TimeBin"))
    .alias("TimeBin")
)

odor_label = [f"Odor{i}" for i in range(1, 6)]
pos_label = [f"Position{i}" for i in range(1, 6)]

# one row per trial
tmp = be_signed.filter(pl.any_horizontal(pl.col(odor_label) == 1))

tmp = tmp.select("TimeBin", *odor_label, *pos_label, "PerformanceLog", "InSeqLog")

tmp = tmp.with_columns(pl.int_range(1, pl.len() + 1).alias("trial_id"))

# which position slot is set
tmp = tmp.with_columns(
    (pl.concat_list(pos_label).list.arg_max() + 1).cast(pl.Int64).alias("pos")
)

tmp = tmp.with_columns(
    pl.col("pos").shift(1, fill_value = pl.col("pos").first() - 1).alias("prev_pos")
)

# boundary 1: position didn't advance by 1
tmp = tmp.with_columns(
    ((pl.col("pos") - pl.col("prev_pos")) != 1).alias("gap")
)

# boundary 2: previous trial was incorrect
tmp = tmp.with_columns(
    ((pl.col("PerformanceLog").shift(1, fill_value = 0)) == -1).alias("err")
)

tmp = tmp.with_columns(
    ((pl.col("gap") | pl.col("err")).cum_sum() + 1).cast(pl.Int64).alias("seq_id")
)

print(tmp["seq_id"].max())
print(tmp["seq_id"])
