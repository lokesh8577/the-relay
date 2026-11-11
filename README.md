# The Relay --- Research Paper Assistant

The Relay is a Python research-paper assistant that retrieves evidence
from a fixed collection of arXiv papers, generates cited answers with an
LLM, validates the response structure, and records latency and
reliability measurements.

## Features

-   Retrieves from exactly three versioned arXiv papers.
-   Extracts PDF text and creates token-based chunks with overlap.
-   Uses BM25 for lexical retrieval.
-   Generates streamed answers through Groq or a local stand-in service.
-   Includes a relevance/answer-validation path, bounded retries, and
    refusal behavior.
-   Simulates slow responses, timeouts, HTTP 429 rate limits, HTTP 503
    failures, flaky recovery, and concurrency pressure.
-   Records raw request-level results and retry-attempt details as CSV
    files.
-   Includes retrieval-quality, answer-quality, and load-test evaluation
    scripts.

## Paper collection

The project uses these exact paper versions:

  -----------------------------------------------------------------------
  Paper                               arXiv version
  ----------------------------------- -----------------------------------
  Attention Is All You Need           `1706.03762v7`

  Moshi: a speech-text foundation     `2410.00037v1`
  model for real-time dialogue        

  Dynamic Chunking                    `2507.07955v2`
  -----------------------------------------------------------------------

The PDFs and derived corpus files are stored under `data/`.

## Project layout

``` text
the-relay/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── paper_downloader.py     # Downloads the pinned arXiv PDF versions
│   ├── chunker.py              # Token-aware chunk creation
│   ├── retrieval.py            # BM25 retrieval and citation formatting
│   ├── groq_client.py          # Groq / stand-in streaming client
│   ├── answer_generator.py     # Answer generation pipeline
│   ├── validator.py            # Structural and citation validation
│   └── reliability.py          # Bounded retry logic
├── standin/
│   └── server.py               # Local OpenAI-compatible stand-in
├── experiments/
│   ├── load_test.py            # Steady, burst, and overload tests
│   ├── summarize_load.py       # Load-test summary generation
│   ├── retry_eval.py           # Retry experiment recorder
│   └── final_metrics.py        # Consolidated result metrics
├── evaluation/
│   ├── questions.json          # Retrieval evaluation questions
│   ├── retrieval_eval.py       # Chunk-setting retrieval evaluation
│   ├── retrieval_audit.py      # Manual retrieval audit
│   ├── answer_eval.py          # Answer-quality evaluation
│   └── summarize_answer_eval.py
├── data/                       # PDFs, extracted text, chunks, manifest
├── results/                    # Raw CSVs and summaries
├── requirements.txt
├── .env                        # Local secrets/configuration (not committed)
└── README.md
```

Some generated data and evaluation output files may be created when the
corresponding scripts run.

## Requirements

-   Python 3.11+ recommended.
-   `pip` and `venv`.
-   A Groq API key only when using the real Groq backend.
-   Internet access to download the arXiv PDFs and call Groq.
-   The local stand-in can be used without a paid service or API key.

The project was developed and tested in a macOS virtual environment.
Other operating systems may require small command adjustments.

## Setup

### 1. Clone or open the project

``` bash
git clone <YOUR_REPOSITORY_URL>
cd the-relay
```

If you already have the project folder, just `cd` into it.

### 2. Create and activate a virtual environment

macOS / Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

``` powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root. Do not commit this file or
share your API key.

Example for the local stand-in:

``` dotenv
MODEL_BACKEND=standin
GROQ_MODEL=openai/gpt-oss-20b
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_TIMEOUT_SECONDS=45
GROQ_MAX_COMPLETION_TOKENS=800

STANDIN_BASE_URL=http://127.0.0.1:8001
STANDIN_BASE_DELAY_SECONDS=0.78
STANDIN_CHUNK_DELAY_SECONDS=0.025
```

For the real Groq backend, change `MODEL_BACKEND` to `groq` and add your
own key:

``` dotenv
MODEL_BACKEND=groq
GROQ_API_KEY=your_groq_api_key
```

Keep `.env` private. The repository should contain `.env.example` with
placeholder values only, never a real key.

## Run the application

Start the research assistant API:

``` bash
uvicorn app.main:app --reload --port 8000
```

Open:

-   `http://127.0.0.1:8000/`
-   `http://127.0.0.1:8000/health`

## Run the local stand-in

In a second terminal, with the virtual environment activated:

``` bash
uvicorn standin.server:app --host 127.0.0.1 --port 8001
```

Check its health endpoint:

``` bash
curl http://127.0.0.1:8001/health
```

Set `MODEL_BACKEND=standin` when running the client against this local
service. The stand-in is intended for controlled experiments; it is not
a production service or a faithful replica of Groq infrastructure.

## Prepare the paper corpus

Download the pinned paper versions:

``` bash
python -m app.paper_downloader
```

Create token-aware chunks using the script's configured/default chunk
settings:

``` bash
python -m app.chunker
```

The resulting extracted-paper and chunk files are stored under `data/`.
Use the exact chunk-size and overlap settings shown by the script and
the generated filenames when reproducing a particular evaluation.

## Run experiments

Keep the stand-in server running on port `8001` for stand-in
experiments.

### Load tests

Examples:

``` bash
python -m experiments.load_test --scenario steady --requests 10 --interval 0.5
python -m experiments.load_test --scenario burst --requests 12
python -m experiments.load_test --scenario overload --requests 20 --interval 0.5
```

The overload scenario caps its arrival interval at 0.05 seconds in the
test harness. The load-test script appends request-level records to
`results/results.csv`.

The quota scenarios (halved RPM and daily-budget exhaustion) were
performed with the stand-in's configurable limits. Check
`standin/server.py` and its environment/configuration before repeating
them; restore the normal values after the experiment.

### Retry experiments

``` bash
MODEL_BACKEND=standin python -m experiments.retry_eval --mode flaky --max-retries 2
MODEL_BACKEND=standin python -m experiments.retry_eval --mode server_error --max-retries 2
MODEL_BACKEND=standin python -m experiments.retry_eval --mode rate_limit --max-retries 2
GROQ_TIMEOUT_SECONDS=2 MODEL_BACKEND=standin python -m experiments.retry_eval --mode timeout --max-retries 1
```

Retry summaries are appended to `results/results.csv`; individual
attempts are recorded in `results/retry_attempts.csv`.

### Retrieval and answer-quality evaluation

The evaluation scripts and question set are under `evaluation/`.

``` bash
python -m evaluation.retrieval_eval
python -m evaluation.retrieval_audit
```

Answer-quality evaluation uses the real model and can consume API quota:

``` bash
python -m evaluation.answer_eval
python -m evaluation.summarize_answer_eval
```

Only run the real answer-evaluation script when you intend to make API
calls. Review the script's options and current evaluation CSV first to
avoid duplicate requests.

### Summaries

``` bash
python -m experiments.summarize_load
python -m experiments.final_metrics
```

## Recorded experimental results

The consolidated `results/results.csv` contains 78 unique request
records at the time of this report:

-   14 real Groq requests.
-   64 stand-in requests.
-   44 successful, 26 error, 7 rate-limited, and 1 timeout outcomes.
-   Nine manually graded answer-quality requests: six correct and three
    wrong.

The stand-in load-test outcomes were:

  Scenario               Requests   Success rate
  -------------------- ---------- --------------
  Steady                       10          70.0%
  Burst concurrency            12         41.67%
  Sustained overload           20          25.0%
  Halved RPM                    8          62.5%
  Daily budget                  8          62.5%

Retrieval audit results (nine questions):

  -----------------------------------------------------------------------
  Chunk size /         Supporting page in top 5 Supporting page at rank 1
  overlap                                       
  ------------------- ------------------------- -------------------------
  300 / 30                                 100%                    55.56%

  500 / 50                                 100%                    66.67%

  700 / 70                                 100%                    77.78%
  -----------------------------------------------------------------------

The answer-quality comparison used only three questions per chunk
setting, so its scores are preliminary and should not be treated as a
broad benchmark. See the engineering report for latency, cost
assumptions, rate-limit evidence, caveats, and detailed retry results.

## Output files

Important generated outputs include:

-   `data/extracted_papers.json` --- extracted paper text and metadata.
-   `data/chunks_*.jsonl` --- token-aware chunks.
-   `results/results.csv` --- consolidated request-level raw results.
-   `results/retry_attempts.csv` --- per-attempt retry history.
-   `results/load_summary.csv` --- load-test summary.
-   `evaluation/answer_results.csv` --- answer-quality grades and
    evaluation metadata.
-   `evaluation/answer_quality_summary.csv` --- answer-quality summary.
-   `The_Relay_Engineering_Report.pdf` --- two-page engineering report
    (place this file in the project root or submission folder).

## Important notes and limitations

-   The local stand-in is a controlled test double. Its quota and
    concurrency behavior should not be interpreted as Groq's actual
    infrastructure.
-   Real Groq tests were modest; no real high-load test was attempted.
-   The answer-quality sample is small (three questions per chunk
    setting) and grades were assigned manually.
-   Latency percentiles from small samples are unstable. They are
    descriptive measurements, not service-level guarantees.
-   Six older baseline records have missing start timestamps; these
    fields were left blank rather than reconstructed.
-   The structural validator checks response shape and citation
    membership, not whether every generated claim is semantically
    entailed by the cited passage.
-   Do not publish `.env`, API keys, or other credentials.

## Reproducibility

For a reproducible run, record:

1.  Python and dependency versions.
2.  The `.env` settings used (excluding secret values).
3.  The exact chunk size, overlap, and retrieval `top_k`.
4.  The stand-in quota/concurrency configuration.
5.  The request CSVs and generated summaries.

The raw CSV files are the source of truth for reported measurements;
summary tables should be regenerated from those records whenever the
dataset changes.
