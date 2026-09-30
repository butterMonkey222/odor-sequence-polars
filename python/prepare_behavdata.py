"""
prepare_behavdata.py: polars port of 1b_prepare_behavdata.R.

Builds, for each rat, the three behaviour tables that 2_extract_spike.R reads:

    All_Odor_Mat   one row per sequence: time, trial, performance, poke and
                   withdraw for each position, the rewards, and ITI6
    Ind_Odor_Mat   one row per InSeq, correct trial, labelled by odour
    Odor5          one row per InSeq position-5 trial: back reward,
                   middle_time, middle_pos and ITI5_end

Verified against the original: on all five rats it reproduces both summary
numbers 1b reports, ITI5 3.623343 and ITI6 4.3515 (means of per-rat means).

The original R and the recordings are not included. The script expects the
behaviour matrices at ../statmatrix/<rat>/*BehaviorMatrix.mat, relative to this
file. Run it directly to process every rat found there.
"""

import math
from pathlib import Path

import polars as pl
from scipy.io import loadmat

DATA = Path(__file__).resolve().parent.parent / "statmatrix"


def gauss_kernel(kernel_size, sigma):
    """Gaussian weights, kernel_size of them, normalised to sum to 1."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    half = kernel_size // 2
    k = [math.exp(-(x**2) / (2 * sigma**2)) for x in range(-half, half + 1)]
    total = sum(k)
    return [v / total for v in k]


def gauss_smooth(col, kernel_size, sigma):
    """Centred Gaussian smoothing of a column, returned as an expression.

    Written as a weighted sum of shifted copies because polars' weighted
    rolling_mean does not accept nulls. A null anywhere in the window, including
    past either edge, makes the sum null, and those rows fall back to the raw
    value: the same result as R's stats::filter followed by 1b's NA fill.
    """
    k = gauss_kernel(kernel_size, sigma)
    half = kernel_size // 2
    smooth = sum(pl.col(col).shift(-j) * k[j + half] for j in range(-half, half + 1))
    return smooth.fill_null(pl.col(col))


# column recipes shared by sections 7 to 9
X = pl.col("XvalRatMazePosition")
Y = pl.col("YvalRatMazePosition")
moving = pl.col("v_x_smooth").abs() > 5


def prepare_rat(path):
    """Run 1b's pipeline on one rat. Returns the three tables script 2 reads."""
    rat = Path(path).parent.name

    # Section 1 load; the column names arrive wrapped three levels deep
    m = loadmat(path)

    col_ids = []
    for ID in m["behavMatrixColIDs"][0]:
        col_ids.append(str(ID[0]))

    be_mat = pl.DataFrame(m["behavMatrix"], schema=col_ids)

    # Section 2 trials and sequences
    # from here on, an OutSeq trial is marked by a negative TimeBin
    be_signed = be_mat.with_columns(
        pl.when(pl.col("InSeqLog") == -1)
          .then(-pl.col("TimeBin"))
          .otherwise(pl.col("TimeBin"))
          .alias("TimeBin")
    )

    odor_label = [f"Odor{i}" for i in range(1, 6)]
    pos_label = [f"Position{i}" for i in range(1, 6)]

    # one row per trial: the bins where an odour fired
    keep = ["TimeBin"] + odor_label + pos_label + ["PerformanceLog", "InSeqLog"]
    tmp = (
        be_signed
        .filter(pl.any_horizontal(pl.col(odor_label) == 1))
        .select(keep)
    )

    # rounded because TimeBin is matched by equality further down
    tmp = tmp.with_columns(pl.col("TimeBin").round(4))
    tmp = tmp.with_columns(pl.int_range(1, pl.len() + 1).alias("trial_id"))

    # position 1-5 from the one-hot columns; arg_max is 0-based and unsigned
    tmp = tmp.with_columns(
        (pl.concat_list(pos_label)
           .list.arg_max()
           .cast(pl.Int64)
           + 1).alias("pos")
    )

    # find_consequtive_seq: a sequence starts where the position did not advance
    # by exactly 1, or right after an incorrect trial. fill_value -1 makes the
    # first trial always a start, so seq_id begins at 1 for every rat, and 1b's
    # "if min(id) == 0" repair is replaced by the assert.
    prev_pos = pl.col("pos").shift(1, fill_value = -1)
    starts_by_pos = (pl.col("pos") - prev_pos) != 1
    starts_by_err = pl.col("PerformanceLog").shift(1, fill_value = 1) == -1
    starts = starts_by_pos | starts_by_err

    seq_id = starts.cum_sum().cast(pl.Int64)

    tmp = tmp.with_columns(
        seq_id.alias("seq_id")
    )
    assert tmp["seq_id"].min() == 1

    # one row per sequence, one column per position. The _Odor<N> names mean
    # position N, not odour N, but fct/get_rest_interval_ensemble_mat.R reads them.
    seq_t = tmp.pivot(
        on = "pos",
        index = "seq_id",
        values = ["TimeBin", "trial_id", "PerformanceLog"],
        sort_columns = True
    )

    vals = ["TimeBin", "trial_id", "PerformanceLog"]

    rename_map = {f"{v}_{i}" : f"{v}_Odor{i}" for v in vals for i in range(1, 6)}

    seq_t = seq_t.rename(rename_map)

    # seq_id onto every 1 ms bin inside a sequence: take the latest sequence
    # start at or before the bin, and keep it only if the bin is also before
    # that sequence's end
    bounds = tmp.group_by("seq_id", maintain_order=True).agg(
        start_t = pl.col("TimeBin").first(),
        end_t = pl.col("TimeBin").last()
    )

    spans = bounds.select(
        "seq_id",
        s = pl.col("start_t").abs(),
        e = pl.col("end_t").abs()
    )

    be_mat = (
        be_mat
        .with_columns(t = pl.col("TimeBin").round(4))
        .join_asof(spans, left_on="t", right_on="s", strategy="backward")
        .with_columns(
            seq_id = pl.when(pl.col("t") <= pl.col("e")).then(pl.col("seq_id")).otherwise(None)
        )
        .drop("t", "s", "e")
    )

    # trial_id by exact time. tmp's OutSeq times are negative, so those trials
    # find no match and stay null, as in 1b.
    be_mat = (
        be_mat
        .with_columns(pl.col("TimeBin").round(4))
        .join(tmp.select("TimeBin", "trial_id"), on="TimeBin", how="left")
    )

    # Section 3 event times, pulled out once so later sections search small tables
    odor_list = {
        nm : be_mat.filter(pl.col(nm) == 1)
                   .select("TimeBin", "seq_id", "trial_id", "PerformanceLog", "InSeqLog")
        for nm in odor_label
    }

    poke = be_mat.filter(pl.col("PokeEvents") == 1).select(poke = "TimeBin")
    withdraw = be_mat.filter(pl.col("PokeEvents") == -1).select(withdraw = "TimeBin")

    FrontReward = be_mat.filter(pl.col("FrontReward") == 1).select("TimeBin")
    BackReward = be_mat.filter(pl.col("BackReward") == 1).select("TimeBin")

    # Section 4 the poke before and the withdraw after each odour onset.
    # join_asof replaces 1b's find_closest_num: backward finds the nearest
    # earlier time, forward the nearest later one. (1b's version excludes an
    # exact tie, join_asof includes it.)
    for nm in odor_label:
        odor_list[nm] = (
            odor_list[nm]
            .join_asof(poke, left_on="TimeBin", right_on="poke", strategy="backward")
            .join_asof(withdraw, left_on="TimeBin", right_on="withdraw", strategy="forward")
            .with_columns(t_diff = pl.col("withdraw") - pl.col("poke"))
        )

    # the same search for each position of all_seq_mat. seq_t's times are signed,
    # so search on abs() and put the sign back: Withdraw_Odor5 > 0 marks InSeq later.
    all_seq_mat = seq_t

    for nm in odor_label:
        sub = (
            all_seq_mat
            .select("seq_id", t = f"TimeBin_{nm}")
            .drop_nulls("t")
            .with_columns(a = pl.col("t").abs())
            .sort("a")
            .join_asof(poke, left_on="a", right_on="poke", strategy="backward")
            .join_asof(withdraw, left_on="a", right_on="withdraw", strategy="forward")
            .select(
                "seq_id",
                (pl.col("t").sign() * pl.col("poke")).alias(f"Poke_{nm}"),
                (pl.col("t").sign() * pl.col("withdraw")).alias(f"Withdraw_{nm}")
            )
        )
        all_seq_mat = all_seq_mat.join(sub, on="seq_id", how="left")

    # Section 5 Ind_Odor_Mat: every trial labelled by odour, then InSeq and
    # correct only. polars has no row names, so the label is added before stacking.
    tmp_train = pl.concat(
        [odor_list[nm].with_columns(Odor = pl.lit(nm)) for nm in odor_label],
        how = "vertical"
    )

    train_testing = tmp_train.filter(
        (pl.col("InSeqLog") == 1) & (pl.col("PerformanceLog") == 1)
    )

    Ind_Odor_Mat = train_testing
    odor_list["Odor5"] = odor_list["Odor5"].filter(pl.col("InSeqLog") == 1)

    # Section 6 the rewards that end a sequence: the first front reward after the
    # position-5 poke, the first back reward after its withdraw. Incorrect trials
    # are dropped first; 1b keeps them and blanks their rewards instead, which
    # gives the same tables because the write-back below joins on seq_id.
    Odor5_Mat = (
        all_seq_mat
        .select("seq_id", "TimeBin_Odor5", "PerformanceLog_Odor5", "Poke_Odor5", "Withdraw_Odor5")
        .drop_nulls()
        .filter((pl.col("TimeBin_Odor5") > 0) & (pl.col("PerformanceLog_Odor5") == 1))
    )

    Odor5_Mat = (
        Odor5_Mat
        .join_asof(FrontReward.select(FrontRewardE = "TimeBin"),
                   left_on="Poke_Odor5", right_on="FrontRewardE", strategy="forward")
        .join_asof(BackReward.select(BackReward = "TimeBin"),
                   left_on="Withdraw_Odor5", right_on="BackReward", strategy="forward")
    )

    # two sequences claiming one back reward: keep the closer one (1b:236-243).
    # Fires once in the dataset, on Buchanan, where the sequence just before a
    # session break has no reward of its own and reaches across the break.
    gap = pl.col("BackReward") - pl.col("Withdraw_Odor5")
    Odor5_Mat = Odor5_Mat.with_columns(
        BackReward = pl.when(gap == gap.min().over("BackReward"))
                       .then(pl.col("BackReward"))
                       .otherwise(None)
    )

    all_seq_mat = all_seq_mat.join(
        Odor5_Mat.select("seq_id", "FrontRewardE", "BackReward"),
        on="seq_id", how="left"
    )

    odor_list["Odor5"] = odor_list["Odor5"].join(
        Odor5_Mat.select("seq_id", "BackReward"),
        on="seq_id", how="left"
    )

    # Section 7 middle_time: when the rat crosses mid-track on its way to the
    # back reward, which splits the pause at the port from the run.
    # X == 0 means no camera frame in that bin.
    run_period = odor_list["Odor5"].select("seq_id", "withdraw", "BackReward").drop_nulls()

    middle_time = []
    middle_pos = []

    for r in range(run_period.height):
        w = be_mat.filter(
            (X != 0)
            & (pl.col("TimeBin") > run_period["withdraw"][r])
            & (pl.col("TimeBin") < run_period["BackReward"][r] + 0.2)
        )

        # velocity in cm/s to the next frame (0.15 cm per position unit);
        # faster than 5 m/s is a tracking glitch
        w = w.with_columns(
            lag = (pl.col("TimeBin").shift(-1) - pl.col("TimeBin")).round(4)
        )

        w = w.with_columns(
            v_x = (3/20) * (X.shift(-1) - X) / pl.col("lag")
        )

        w = w.with_columns(
            v_x = pl.when(pl.col("v_x").abs() > 500).then(None).otherwise(pl.col("v_x"))
        )

        w = w.with_columns(v_x_smooth = gauss_smooth("v_x", 51, 10))

        # mid-track and moving; the frame closest to X = 500
        mid = w.filter((X > 250) & (X < 800) & moving)

        if mid.height != 0:
            dist = (mid["XvalRatMazePosition"] - 500).abs()
            pos_mid = mid["XvalRatMazePosition"][dist.arg_min()]
            middle_pos.append(pos_mid)
            middle_time.append(mid.filter(X == pos_mid)["TimeBin"][0])
        else:
            # 1b's fallback; fires on four of Barat's runs and nowhere else
            print(f"{rat}: middle_time fallback, run {r + 1}")
            est = w.filter((X > 200) & moving)
            dist = (est["XvalRatMazePosition"] - 800).abs()
            pos_last = est["XvalRatMazePosition"][dist.arg_min()]
            middle_pos.append(est["XvalRatMazePosition"].mean())
            middle_time.append(min(est.filter(X == pos_last)["TimeBin"][0] - 0.2,
                                   run_period["BackReward"][r]))

    run_period = run_period.with_columns(
        middle_time = pl.Series(middle_time),
        middle_pos = pl.Series(middle_pos)
    )

    odor_list["Odor5"] = odor_list["Odor5"].join(
        run_period.select("seq_id", "middle_time", "middle_pos"), on="seq_id", how="left"
    )

    # Section 8 ITI5: from the position-5 withdraw until the rat stops pausing
    # at the port, searched inside withdraw -> middle_time
    run_period = odor_list["Odor5"].select("seq_id", "withdraw", "middle_time", "BackReward").drop_nulls()

    ITI5_end = []

    for r in range(run_period.height):
        p = be_mat.filter(
            (X != 0)
            & (pl.col("TimeBin") >= run_period["withdraw"][r])
            & (pl.col("TimeBin") <= run_period["middle_time"][r])
        )

        p = p.with_columns(
            lag = (pl.col("TimeBin").shift(-1) - pl.col("TimeBin")).round(4)
        )

        p = p.with_columns(
            v_x = (3/20) * (X.shift(-1) - X) / pl.col("lag")
        )

        p = p.with_columns(
            v_x = pl.when(pl.col("v_x").abs() > 500).then(None).otherwise(pl.col("v_x"))
        )

        p = p.with_columns(v_x_smooth = gauss_smooth("v_x", 51, 10))

        # samples at the port; t_lag is seconds since withdraw
        s = (
            p.filter((X > 0) & (X < 200) & (Y > 200) & (Y < 450))
            .select("TimeBin", "XvalRatMazePosition", "YvalRatMazePosition", "v_x_smooth")
            .with_columns(t_lag = pl.col("TimeBin") - run_period["withdraw"][r])
            .drop_nulls()
        )

        # number the still stretches. Signed > 5, not abs: only forward motion
        # counts as leaving the port.
        s = s.with_columns(speed_id = (pl.col("v_x_smooth") > 5).cum_sum())
        s = s.with_columns(run_length = pl.len().over("speed_id"))

        q = s.filter((pl.col("run_length") != 1) & (pl.col("t_lag") > 1) & (pl.col("t_lag") < 4))

        if q.height != 0:
            # the pause ends at the last row of the first unbroken block
            first = q.filter(pl.col("run_length").rle_id() == 0)
            ITI5_end.append(first["TimeBin"][-1])
        else:
            # 1b's fallback: the first sample past 1 s. Fires nowhere in this dataset.
            print(f"{rat}: ITI5 fallback, run {r + 1}")
            ITI5_end.append(s.filter(pl.col("t_lag") > 1)["TimeBin"][0])

    run_period = run_period.with_columns(ITI5_end = pl.Series(ITI5_end))
    odor_list["Odor5"] = odor_list["Odor5"].join(
        run_period.select("seq_id", "ITI5_end"), on="seq_id", how="left"
    )

    # Section 9 ITI6: the first sustained still stretch at the back of the track,
    # after the back reward and before the next sequence starts.
    # next_poke is the NEXT sequence's Poke_Odor1, shifted over all rows before
    # filtering, as 1b's lead() does.
    rp6 = (
        all_seq_mat
        .select("seq_id", "Withdraw_Odor5", "FrontRewardE", "BackReward",
                next_poke = pl.col("Poke_Odor1").shift(-1))
        .drop_nulls(["Withdraw_Odor5", "BackReward"])
        .filter(pl.col("Withdraw_Odor5") > 0)
    )

    # stands in when there is no next poke; mean() skips nulls
    avg_stay_back = (rp6["next_poke"] - rp6["BackReward"]).mean()

    rp6 = rp6.with_columns(
        upper = pl.when(pl.col("next_poke").is_null())
                  .then(pl.col("BackReward") + avg_stay_back)
                  .otherwise(pl.col("next_poke")),
        lower = pl.when(pl.col("next_poke").is_not_null() & (pl.col("next_poke") - pl.col("BackReward") < 5))
                  .then(pl.col("FrontRewardE"))
                  .otherwise(pl.col("BackReward") - 1)
    )

    # unlike sections 7 and 8, 1b computes this velocity over the whole session
    # and smooths it before windowing, with no glitch filter, rounded to 2 dp
    sm = be_mat.filter(X != 0).select("TimeBin", "XvalRatMazePosition", "YvalRatMazePosition")
    sm = sm.with_columns(lag = (pl.col("TimeBin").shift(-1) - pl.col("TimeBin")).round(4))
    sm = sm.with_columns(v_x = (3/20) * (X.shift(-1) - X) / pl.col("lag"))
    sm = sm.with_columns(v_x_smooth = gauss_smooth("v_x", 51, 10).round(2))

    ITI6_start = []
    ITI6_end = []

    for r in range(rp6.height):
        k = sm.filter(
            (pl.col("TimeBin") > rp6["lower"][r]) & (pl.col("TimeBin") < rp6["upper"][r])
            & (X >= 750) & (X <= 1200)
        )
        # abs here: at the back, movement either way ends a still stretch
        k = k.with_columns(speed_id = (pl.col("v_x_smooth").abs() > 5).cum_sum())
        k = k.with_columns(run_length = pl.len().over("speed_id"))
        cand = k.filter(pl.col("run_length") != 1)
        ITI6_start.append(cand["TimeBin"][0])
        first6 = cand.filter(pl.col("run_length").rle_id() == 0)
        ITI6_end.append(first6["TimeBin"][-1])

    rp6 = rp6.with_columns(ITI6_start = pl.Series(ITI6_start), ITI6_end = pl.Series(ITI6_end))
    all_seq_mat = all_seq_mat.join(
        rp6.select("seq_id", "ITI6_start", "ITI6_end"),
        on="seq_id", how="left"
    )

    return {
        "All_Odor_Mat": all_seq_mat,
        "Ind_Odor_Mat": Ind_Odor_Mat,
        "Odor5": odor_list["Odor5"],
    }


if __name__ == "__main__":
    be_data = {}
    for f in sorted(DATA.glob("*/*BehaviorMatrix.mat")):
        be_data[f.parent.name] = prepare_rat(f)
    if not be_data:
        raise FileNotFoundError(f"no behaviour matrices under {DATA}")

    for rat, tables in be_data.items():
        print(f"{rat:11s}", "  ".join(f"{name} {df.shape}" for name, df in tables.items()))

    # 1b's own check: the mean of the per-rat means, over all five rats
    iti5 = [(d["Odor5"]["ITI5_end"] - d["Odor5"]["withdraw"]).mean() for d in be_data.values()]
    iti6 = [(d["All_Odor_Mat"]["ITI6_end"] - d["All_Odor_Mat"]["ITI6_start"]).mean() for d in be_data.values()]
    print(f"ITI5 mean of per-rat means {sum(iti5) / len(iti5):.6f}   1b reports 3.623343")
    print(f"ITI6 mean of per-rat means {sum(iti6) / len(iti6):.6f}   1b reports 4.3515")
