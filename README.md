# Odor sequence pipeline — polars port

A Python/polars port of `1b_prepare_behavdata.R`, the behaviour preprocessing
step of the Fortin Lab pipeline for hippocampal tetrode recordings from rats
performing an odor sequence memory task
([Shahbaba et al. 2022](https://www.nature.com/articles/s41467-022-28057-6),
UC Irvine).

The port is complete and verified against the original. It is one file,
`prepare_behavdata.py`. The original R is not mine and isn't included here.

## What it builds

For each rat, `prepare_rat(path)` turns the 3.7 million 1 ms bins of the
behaviour matrix into the three tables that `2_extract_spike.R` reads:

| table | one row per | holds |
|---|---|---|
| `All_Odor_Mat` | sequence | time, trial, performance, poke and withdraw at each position; the front and back rewards; ITI6 start and end |
| `Ind_Odor_Mat` | in-sequence, correct trial | time, trial, performance, poke, withdraw and hold time, labelled by odor |
| `Odor5` | in-sequence trial at position 5 | back reward, mid-track crossing (`middle_time`, `middle_pos`), ITI5 end |

ITI5 is the pause at the odor port after the last odor, before the rat runs to
the back of the maze. ITI6 is the still period at the back after the reward,
before it returns for the next sequence.

## Verification

On all five rats it reproduces both summary statistics recorded in `1b`:
ITI5 = 3.623343 s and ITI6 = 4.3515 s, each the mean of the per-rat means. Both
depend on every earlier step, so together they check the whole pipeline. Each
step was also checked against a step-by-step R replication on one rat (Barat).
The tables have not yet been compared cell by cell with the saved `.RData`.

## Differences from 1b

- `Odor5` has no `FrontReward` column. `1b` adds one to the odor 5 table in its
  poke and withdraw step, but nothing downstream reads its value.
- Incorrect trials are dropped before the reward search instead of blanked
  after it. The tables come out the same, because results are written back by
  joining on `seq_id` rather than by row position.
- Nearest-time searches use as-of joins, which count an exact tie as a match
  where `find_closest_num` does not.

## Implementation notes

- The per-trial search loops become as-of joins (`join_asof`), so the whole
  pipeline runs in about a second per rat.
- Gaussian smoothing is written as a weighted sum of shifted columns, because
  polars' weighted `rolling_mean` rejects nulls. It matches R's `stats::filter`,
  including its handling of missing values, to within 1e-13.
- Known quirks of `1b` are kept so that the output matches it, and are commented
  in the code where they occur.
- The tables are returned in memory as polars DataFrames. Nothing is written to
  disk yet.

## How sequences are defined

Trials arrive as 3.7 million 1 ms bins with one-hot odor and position columns.
The task is to group them into *sequences* — continuous runs through the five
positions of the task.

A sequence ends in exactly two ways, so there are two boundary rules:

1. the position didn't advance by exactly 1
2. the previous trial was incorrect, which terminates the sequence early

A running count of boundaries gives the sequence number.

Rule 1 also splits on trials deleted from the session during preprocessing, so a
single real attempt can appear as two sequences. That's intended: if a position
was never observed, a continuous run through it can't be claimed. A "sequence"
here means an observed continuous run, not an attempt.

Verified against the R on Barat: 176 trials, 21 out-of-sequence, 49 gaps,
11 errors, 50 sequences. Sequence lengths sum to 176 and none exceeds 5.

## Running it

```
pip install polars scipy
python prepare_behavdata.py
```

Processes every rat it finds at `statmatrix/<rat>/*BehaviorMatrix.mat`, in the
folder next to the script, then prints each rat's table shapes and the two
summary statistics. The data isn't public, so this won't run without it.

To use it from other code, from the repository root:

```python
from prepare_behavdata import prepare_rat

tables = prepare_rat("statmatrix/Barat/Barat-11-06-2008Skips_mrg_GEcut_SGselected_BehaviorMatrix.mat")
tables["All_Odor_Mat"]    # also "Ind_Odor_Mat" and "Odor5"
```
