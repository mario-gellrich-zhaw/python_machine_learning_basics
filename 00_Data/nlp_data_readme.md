# Data for the HS2026 NLP notebooks

## Word-map activity

The word-map notebooks train word2vec using the supplied email text, then let
students choose which learned words to plot. All vectors come from that trained
model; no separate word-vector CSV is needed. The map on page 17 of
`Slides_MSC_Data_Science_Week_08_EN.pdf` is an example, not a required result.

Students edit `chosen_words`, check cosine similarities using all 100 values
per word, and use PCA to draw a two-dimensional view. Words not learned during
training are reported and left out; at least three known words are needed.
Changing the selection does not require retraining the model: rerun sections
6–9 to calculate a new similarity table and map.

The percentage of variation kept depends on the selected words. It is not a
measure of language understanding or spam accuracy. Do not copy the full
original word-vector cache into the repository.

## Authorised email text for word2vec training

`email_spam_text.csv.gz` contains the 30,000 selected emails from the original
page-17 word2vec experiment, not the full original corpus. Source: the local copy
of [Spam email dataset (Enron, TREC and others)](https://www.kaggle.com/datasets/nikhilchaudhary7108/spam-email-datasetenron-trec-and-others).
The instructor confirmed redistribution permission for this corpus and the
repository's intended audience on 2026-11-02. This confirmation is not an
independently audited upstream licence or permission for unrestricted reuse.

Columns:

- `message`: original email text, potentially including subjects, bodies,
  headers and personal information; it is not anonymized.
- `label`: `0 = ham`, `1 = spam`.
- `split`: fixed `train` (18,000), `validation` (6,000) or `test` (6,000).
- `source_row`: row in the original local CSV, for provenance only.
- `message_key`: SHA-256 of normalized decoded subject/body text, for integrity
  and duplicate checks; hashing does not anonymize the message.

The export preserves the original selected message text, labels, partition
membership and order within each partition. Decoded training tokens and their
order match the input to the original page-17 run. The shared Neural Networks
loader verifies the stored keys and preserves these partitions rather than
resampling. Changing the seed does not resplit this prepared text file.
Do not lower `MAX_MESSAGES` below 30,000 for this fixed dataset.

The separate `email_spam.csv` contains only numerical features and has its own
reference identifiers. It remains unchanged; do not assume its source-row
identifiers and message keys are interchangeable with this word2vec subset.

Pandas reads the gzip CSV directly. The word-map notebooks use it by default,
so no extra upload or download is required in Codespaces. `RAW_EMAIL_PATH` or
`SPAM_DATA_PATH` can still select another authorised raw CSV with `message` and
`label` columns; that changes the experiment. Keep additional private data
outside version control, for example in the ignored `00_Data/private/` directory.
Do not print raw messages or publish the full learned vocabulary. The notebook
keeps the trained model in memory and does not save a checkpoint automatically.

Only the training partition supplies word2vec pairs; validation/test text is not
used to learn vectors. This static random, stratified split is not a substitute
for chronological evaluation of a deployed spam filter. Original five-epoch
training used NumPy 2.5.4 and PyTorch 2.14.1; different packages, text or settings
can change the vectors even with the same seed. Report the results from the
model and words you actually used.

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
dataset is assumed. Do not add the full private Parquet file or additional
unapproved raw email data.
