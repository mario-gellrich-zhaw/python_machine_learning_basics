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
one worked step for the lecture's ten-minute demonstration; allow separate
practical time for the complete exercises. No Python setup or general EDA is
repeated.

Exercises contain a few `None` values marked `# TODO`. Replace these before
running the affected cells. Write short answers in the marked Markdown cells.
Solutions have complete code and answer guidance. Code checks belong outside
the student notebooks; there are no `assert` statements in them.

## Running in Codespaces

The existing container uses Python 3.11 and CPU-only PyTorch 2.2.2. The root
requirements now also install Sentence Transformers 2.7.0 and SentencePiece
0.2.0, alongside the existing pinned Transformers 4.40.2 environment.
Rebuild an existing Codespace after pulling these changes and select its
Python 3.11 kernel. No GPU, paid service, API key or private dataset is required.

Run each notebook with its lecture folder as the working directory. Data paths
are `../../../00_Data/`. First use of BERT or E5 downloads public model files
to the Hugging Face cache. Text is processed locally, not sent to a hosted
prediction API. Prepare downloads before class; first-time download and encoding
time are not part of a ten-minute demonstration. The BERT exercise downloads a
roughly 700 MB multilingual model. Models are used sequentially on CPU.

## Data and interpretation

- [nlp_teaching_jobs.csv](../../00_Data/nlp_teaching_jobs.csv) contains original
  synthetic job advertisements with dates, four categories and fixed chronological
  training/validation/test membership. These are small teaching examples, not
  the real collection or evidence of deployment performance.
- [nlp_search_benchmark.json](../../00_Data/nlp_search_benchmark.json) preserves
  the 24 approved, short German job-ad excerpts, anonymized labels and draft
  query judgments from the existing NLP lecture. Only fragments are indexed.
  Judgments need instructor review. The revised query panel is not untouched
  holdout evaluation, and recall applies only to the 24 excerpts.
- Bandit clicks use invented 8%/12% probabilities. There are no real user logs.

See the [data notes](../../00_Data/nlp_data_readme.md). The private 30,000-ad
Parquet file and raw emails are deliberately not copied into this repository.
The exercises do not reproduce the numerical results from the real-data slides.

## Reused lecture material

Adapted from the existing `NLP` word2vec/job-category and BERT/job-category
notebooks, the hybrid search notebook and its transparent BM25/RRF implementation,
and the current two-ad bandit simulator. The short word2vec trainer uses the
full-softmax teaching example instead of the larger notebook's negative-sampling
implementation. The search exercise compares BM25, E5 and RRF; plain BERT search
remains an optional extension. RLHF is a conceptual activity, not a training run.

Primary references and official documentation are linked in each notebook.
