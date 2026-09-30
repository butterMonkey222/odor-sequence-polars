# Odor sequence pipeline — polars port

A polars port of `1b_prepare_behavdata.R`. For each rat, `prepare_rat(path)`
returns the three tables `2_extract_spike.R` reads: `All_Odor_Mat`,
`Ind_Odor_Mat` and `Odor5`.

**Check:** on all five rats it reproduces both numbers recorded in `1b`,
ITI5 = 3.623343 and ITI6 = 4.3515 (means of the per-rat means).

**Known difference from `1b`:** `Odor5` has no `FrontReward` column, which
nothing downstream reads. The tables have not been compared cell by cell with the
saved `.RData`.

The original R and the data belong to the Fortin Lab and aren't included.

## Running it

```
pip install polars scipy
python prepare_behavdata.py
```

It expects the behaviour matrices at `statmatrix/<rat>/*BehaviorMatrix.mat`,
next to the script, and prints each rat's table shapes and the two numbers above.
