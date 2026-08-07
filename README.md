# Odor sequence pipeline — polars port

Porting `1b_prepare_behavdata.R` from R to Python/polars. Hippocampal tetrode
recordings from rats performing an odor sequence memory task
([Shahbaba et al. 2022](https://www.nature.com/articles/s41467-022-28057-6),
Fortin Lab, UC Irvine).

Work in progress. The original R is not mine and isn't included here.

## Contents

| file | what it does |
|---|---|
| `python/find_consequtive_seq_Barat.py` | assigns a sequence number to each trial |

## find_consequtive_seq

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
cd python
python find_consequtive_seq_Barat.py
```

Expects the `.mat` behavior matrix at `../statmatrix/<rat>/`. The data isn't
public, so this won't run without it.
