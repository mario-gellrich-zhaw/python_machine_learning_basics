# Data for the HS2026 NLP notebooks

## Synthetic classification examples

`nlp_teaching_jobs.csv` is an original, generated teaching dataset. It contains
120 short German advertisements in four categories. The dates and labels are
constructed. Fixed partitions contain 72 training, 24 validation and 24 test
rows, ordered by date. Texts are unique, but share simple category-specific
templates across partitions. This makes the exercise easier than real job
classification; it does not test robustness to new writing styles or markets.
Do not present its metrics as real business evidence.

No employer details, original ad IDs or private descriptions are included.
The generator uses seed 7; the CSV is supplied so students need not regenerate it.

## Approved search excerpts

`nlp_search_benchmark.json` is copied from the existing Lecture 3 benchmark.
It contains only 24 approved short redacted excerpts, teaching IDs, anonymized
display labels, queries and provisional judgments. It is not the full private
job-ad collection and is not the older Kaggle LinkedIn dataset.

The supplied source metadata, omission markers and judgment status are retained.
Titles are not original job titles and are not searched. Judgments are assistant
drafts awaiting instructor review. T03 was revised after earlier results were
inspected, so these results are a revised teaching comparison, not untouched
holdout performance. Missing excerpt evidence does not prove whole-ad irrelevance.

Teaching permission for these short excerpts does not establish an unrestricted
public-data licence. Confirm the intended distribution permissions before making
a public fork or redistributing this data. No licence from the unrelated Kaggle
dataset is assumed. Do not add the full private Parquet file or raw emails.
