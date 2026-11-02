# Data

`train.py` expects `data/final_combined_data.csv`: Global Findex 2021 individual
responses for Kenya, Tanzania and Uganda (1,000 each), combined with country-level
mobile money indicators.

Codes (Global Findex 2021 codebook):
- `female`: 1 = Female, 2 = Male
- `urbanicity`: 1 = Rural, 2 = Urban
- `educ`: 1 = Primary or less, 2 = Secondary, 3 = Tertiary or more
- `inc_q`: 1 = poorest 20% ... 5 = richest 20% (within country)
- `dig_account` (target): 1 = has a digital financial account

The CSV is not committed yet (see .gitignore) until its licence terms are confirmed.
