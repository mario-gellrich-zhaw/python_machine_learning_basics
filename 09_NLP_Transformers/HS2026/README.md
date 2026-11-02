# NLP and Transformers - HS2026

Each lecture follows the Neural Networks notebooks: read the concept notebook,
complete the exercise, then compare with the solution.

| Lecture | Topic | Notebook prefix |
|---|---|---|
| [1](lecture1/) | Words, vectors, word2vec and cosine | `text_embeddings` |
| [2](lecture2/) | Self-attention, BERT and text generation | `transformers` |
| [3](lecture3/) | Job search, embeddings, BM25 and top-K evaluation | `embedding_search` |
| [4](lecture4/) | Feedback, epsilon-greedy, sequential decisions and RLHF | `learning_from_feedback` |

Each folder contains `<prefix>_concept.ipynb`, `<prefix>_exercise.ipynb` and
`<prefix>_solution.ipynb`. The concepts provide short worked examples. Choose
one worked step for a lecture demonstration; allow separate
practical time for the complete exercises. No Python setup or general EDA is
repeated.

Lecture 1's [concept notebook](lecture1/text_embeddings_concept.ipynb) restores
the spam example from slide 11: `claim prize` becomes `[0.5, 1.5]`, then a
separate classifier estimates a spam probability for a review decision. Its
small synthetic message set compares an always-normal baseline, TF-IDF and
invented word-vector means, with validation-selected review thresholds and
held-out evaluation. This is an illustrative worked example, not a trained
word2vec spam benchmark. The job-category exercise remains separate practical
work and learns word2vec from its training text.

To train word2vec and explore a word map like the example on page 17 of
`Slides_MSC_Data_Science_Week_08_EN.pdf`, use the additional
[word-map exercise](lecture1/word_relationships_exercise.ipynb) and
[solution](lecture1/word_relationships_solution.ipynb). Students now train
word2vec on an authorised copy of the original email corpus: form word/context
pairs, update two embedding tables with negative sampling, inspect loss,
then choose their own words, check cosine similarities, and plot the learned
vectors with PCA. Edit `chosen_words` in section 6 and rerun sections 6–9 to
explore another selection without retraining. Unknown words are reported and
left out; at least three known words are needed.
The training loop is visible, with an optimizer-update gap in the exercise.
Its live loss curve updates every 500 batches and at each epoch's end, showing
the current epoch, total word pairs processed and recent mean training loss.
This measures the word-pair learning task, not spam accuracy or held-out performance.
The existing Neural Networks email loader is reused. Training uses only training
emails; spam labels are not part of the word2vec objective.

The default dataset is the included
[email_spam_text.csv.gz](../../00_Data/email_spam_text.csv.gz): the authorised
30,000-email text subset from the original word2vec experiment, with fixed
18,000/6,000/6,000 training/validation/test partitions and original row order.
Pandas reads it directly without extraction, additional uploads or a Kaggle
account. The instructor confirmed permission for this corpus and the intended
repository audience; this is not a blanket redistribution licence.
The feature-only `email_spam.csv` cannot train word2vec and has its own reference
metadata. No checkpoint is saved automatically.

`RAW_EMAIL_PATH` or `SPAM_DATA_PATH` can still point to another authorised raw
CSV with `message` and `label` columns. Additional private data belongs outside
Git, for example in the ignored `00_Data/private/` directory. Do not print raw
messages, publish the full vocabulary or add other private corpora.

No separate saved word-vector CSV is used. The plot always uses the model
trained in the notebook and the student's chosen words. The percentage of
variation kept changes with that selection; it is not a target to match.
A short lecture demonstration can show training progress and the completed
graph; keep the full training and PCA activity for practical work.

Exercises contain a few `None` values marked `# TODO`. Replace these before
running the affected cells. Write short answers in the marked Markdown cells.
Solutions have complete code and answer guidance. Code checks belong outside
the student notebooks; there are no `assert` statements in them.

## Running in Codespaces

The existing container uses Python 3.11 and CPU-only PyTorch 2.2.2. The root
requirements now also install Sentence Transformers 2.7.0 and SentencePiece
0.2.0, alongside the existing pinned Transformers 4.40.2 environment.
Rebuild an existing Codespace after pulling these changes and select its
Python 3.11 kernel. No GPU, paid service, API key or separate email upload is
required. The word-map training activity uses the supplied compressed text subset.

Run each notebook with its lecture folder as the working directory. Data paths
are `../../../00_Data/`. First use of BERT or E5 downloads public model files
to the Hugging Face cache. Text is processed locally, not sent to a hosted
prediction API. Prepare downloads before class; first-time download and encoding
time should be allowed for separately. The BERT exercise downloads a
roughly 700 MB multilingual model. Models are used sequentially on CPU.

## Data and interpretation

- [nlp_teaching_jobs.csv](../../../00_Data/nlp_teaching_jobs.csv) contains original
  synthetic job advertisements with dates, four categories and fixed chronological
  training/validation/test membership for Lecture 1. These are small teaching examples, not
  the real collection or evidence of deployment performance.
- [nlp_real_job_categories.csv.gz](../../../00_Data/nlp_real_job_categories.csv.gz)
  supplies Lecture 2's exact real-data cohort from Week 08, page 37: 5,011 cleaned,
  redacted descriptions across 143 categories, split into 3,328 training, 918
  validation and 765 test ads by date. Later unseen categories remain in evaluation.
  The notebooks compare frequency, TF-IDF and frozen multilingual BERT using the
  lecture recipe and report accuracy, macro-F1 and top-3 hit rate. Full CPU encoding
  is practical work; prepare the solution outputs before a short lecture demo.
  The instructor confirmed permission for this subset and repository audience,
  not unrestricted reuse of the full source collection.
- [nlp_search_benchmark.json](../../../00_Data/nlp_search_benchmark.json) preserves
  the 24 approved, short German job-ad excerpts, anonymized labels and draft
  query judgments from the existing NLP lecture. Only fragments are indexed.
  Judgments need instructor review. The revised query panel is not untouched
  holdout evaluation, and recall applies only to the 24 excerpts.
- Lecture 3's additional **Search the larger real-ad collection** section reuses
  all 5,011 ads from Lecture 2 for interactive BM25, E5 and hybrid search.
  Edit `user_query` and rerun its code cell to compare top-K lists. E5 ad vectors
  are encoded in batches with progress, then cached in the Git-ignored
  `00_Data/nlp-search-cache/` directory. Cache matching includes the dataset,
  model revision, input limit and package versions; changes trigger a reported
  rebuild. A corrupt cache raises an error rather than silently giving results.
  BM25 searches full cleaned descriptions; E5 uses at most 512 tokens, including
  its prefix and special markers. The notebook reports how many ads exceed the
  actual model limit. No full-ad precision/recall is calculated: the excerpt
  judgments do not cover this larger collection. Prepare encoding before the
  lecture; use one query comparison as the short demonstration.
- Bandit clicks use invented 8%/12% probabilities. There are no real user logs.
- [email_spam_text.csv.gz](../../../00_Data/email_spam_text.csv.gz) supplies the
  selected original email texts for word2vec training. Preserve its partitions
  and row order; fit word2vec only on its training text. See its distribution
  and privacy notes before reuse.

See the [data notes](../../../00_Data/nlp_data_readme.md). The private 30,000-ad
Parquet file and the full original email corpus are not copied into this repository.
Only the authorised classification and email subsets described above are included.
Lecture 1's synthetic job exercise does not reproduce real-data slide metrics.
Lecture 2 now uses the original real classification cohort and recipe; report
the results of your actual run rather than assume identical numbers across
package versions. The word-map activity uses page 17 as an illustration and
encourages students to explore other words rather than copy a fixed result.

## Reused lecture material

Adapted from the existing `NLP` word2vec/job-category and BERT/job-category
notebooks, the hybrid search notebook and its transparent BM25/RRF implementation,
and the current two-ad bandit simulator. The short word2vec trainer uses the
full-softmax teaching example instead of the larger notebook's negative-sampling
implementation. The additional word-map notebooks reuse the original spam
notebook's negative-sampling recipe and train it on authorised email text.
The search exercise compares BM25, E5 and RRF; plain BERT search
remains an optional extension. RLHF is a conceptual activity, not a training run.

Primary references and official documentation are linked in each notebook.
