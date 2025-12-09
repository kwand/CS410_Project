# CS410 Course Project - Software Toolkit for Sentiment Analysis of Music Competitions

Authors: Daniel Kwan (dwkwan2)

This software toolkit enables **sentiment analysis of the YouTube audience in multi-stage (classical) music competitions** and **comparing the public sentiment
against the official scores given by a professional jury**. While we build and demonstrate our tooklkit around the classical music setting, our code is generic enough 
and configurable to handle any kind of music competition, provided YouTube comments and jury scores (in a specific format are available).

At a high level, our toolkit can:
- automatically **scrape YT comments** from matching video recordings of competition performances
- **preprocess scrapped comments** before passing into the sentiment analysis pipeline (i.e. remove comments published after official results were annouced, ignore replies, etc.)
- **analyze comment sentiment** by assigning scores to each preprocessed comment, with multiple backends available including:
    - traditional NLP methods (NTLK's VADER)
    - multilingual-capable, LM-based methods (HuggingFace Transformers and XLM-RoBERTa)
    - custom method using the latest LLMs (Ollama and `qwen3:0.6b`) - *warning: compute-heavy and may take up to 6hrs on large datasets (such as the current config)*
- **generate reports** interpretting the sentiment-scored comments and aggregating them into a single score per YT video (i.e. competition performance). Currently, we can output:
    - **per-stage rankings** of competitors according to our sentiment-derived scores, and comparing against the official rankings by computing **ranking metrics** (including precision @ k, recall, and nDCG)
    - **correlations** and correlation plots of sentiment scores against the official jury scores
    - **score distribution** plots of our sentiment scores against the official jury scores

We test our toolkit on the most recent 2025 International Chopin Competition (Oct 2 - Oct 20), which aside from being relatively-transparent music competition with all official jury scores publicly-released, it
was also followed quite closely by a multilingual YouTube audience (English, Polish, Japanese, Chinese, etc.) - making it an excellent test subject.

## Installation

The code was developed and tested in a conda environment with Python 3.12. We recommend creating a similar environment, then installing the requirements

```
pip install -r requirements.txt
```

Installing VADER requires an extra manual step:
```
python -m nltk.downloader vader_lexicon
```

For using Ollama as the sentiment analysis backend, please refer to the [official Ollama docs](docs.ollama.com) for installation and to start up the server i.e. `ollama serve` before running our code.


## How-to-use

### Required Inputs

To use this toolkit **on a different config than the one currently supplied (for the 2025 Chopin)**, please provide the following:

- **a config YAML file** (placed under `config/`) similar to `config/chopin2025_config.yaml`. In particular, the YT channel URL, regex pattern for matching videos, and path to official scoring data must be provided. (Remember to pass the `--config <path_to_your_config>` flag to each of the commands)

```yaml
api:
  # API key is best supplied via env var; fallback to literal key if desired.
  key_env: "YOUTUBE_API_KEY"
  key: ""

scrape:
  # Max number of videos to consider (-1 means no limit).
  max_videos: -1
  # Max comments per video (-1 means no limit).
  max_comments: -1
  include_replies: true
  debug_log_unmatched: true
  debug_log_path: "data/chopin_2025/unmatched_videos.log"
  debug_log_all_videos: true
  debug_log_all_path: "data/chopin_2025/all_channel_videos.log"


paths:
  # Base data directory for raw/processed artifacts.
  data_dir: "data"

competition:
  name: "Chopin 2025"
  channel:
    # YouTube channel URL (handle form preferred, e.g., https://www.youtube.com/@chopininstitute).
    url: "https://www.youtube.com/@chopininstitute"
    # Regex or plain substring to match video titles.
    keyword_pattern: "(19th Chopin Competition, Warsaw)"
  # Pattern for parsing video titles into (competitor name, round).
  title_pattern: "^([A-Z][A-Z\\s]+?)\\s+–\\s+([a-z]+) round"
  round_to_stage:
    first: "stage1"
    second: "stage2"
    third: "stage3"
    final: "stage4"
  average_score_key: "cmean"
  official_scores_path: "raw_data/official_scores.json"
  # Date filters for preprocessing (inclusive).
  date_filter_start: "2025-10-02"
  date_filter_end: "2025-10-20"
  ignore_replies_for_processing: true
```

- **official scoring data in JSON format** similar to `raw_data/official_scores.json`. It should have at least the following minimal structure, although additional keys are allowed to be present (as in the current json, but will not be used):

```
[
    {
        "name": "Smith, John", // Competitor name in last, first format
        "stage1": {
            "mean": 20.88, // "mean" or the score key label as specified in competition.average_score_key of the config.yaml
            "scores": { // individual jury member scores, used in correlation reporting
                "DS": 15, 
                "GO": 22,
                "AE": 22,
                ...
            },
            "result": true // whether the competitor advanced to the next stage
        },
        ...
        "stage4": {
            "mean": 20.88,
            "scores": {
                "DS": 19,
                ...
            },
            // The "result" key for the final competition stage stores the integer ranking if they placed 
            // (in this case, "3" -> awarded third place)
            // Otherwise, "false" if they did not advance to the awards podium
            "result": 3 
        },
    }, 
    // And so on, for each competitor...
    {
        "name": "McDonald, Ronald", 
        ...
    }
    ...
]
```

### Please use `--help` to see all available parameters for each command

Our code is built using the `click` CLI library, which provides detailed help messages and easy valid-input checking.

### Step 1 - Scrapping + preprocessing comments

**You must first acquire an YouTube Data API key** and specify it in the YAML config (under `api.key`) or otherwise in a `YOUTUBE_API_KEY` env variable. See [Google's official guide](https://developers.google.com/youtube/v3/getting-started) for more details.

Google enforces weekly limits to the number of API credits - this step will consume many credits although we already attempt to minimize the number of requests. If you wish to perform sentiment analysis
on the provided config (Chopin 2025) only, **scrapped and preprocessed data** is already provided under `data/chopin_2025/` and will be automatically detected by subsequent steps (if left at its current path).
```
python main.py process_data 

# Optionally, provide --data-dir to override output dir
```
By default writes raw to `data/<dataset>/scrapped_data/comments_raw.jsonl` and filtered to `data/<dataset>/processed_data/comments_processed.jsonl` (see global variables in `main.py` if you wish to change these) where dataset is determined by the `competition.name` key in the config YAML.

This step will scrape all comments from matching YT videos using the provided regex and preprocess them according to the filtering rules in the config (such as filtering by the date range specified in `date_filter_start` and `date_filter_end` to avoid our results being biased by comments posted after the official results were known i.e. people are more likely to praise known winners of a competition). 

**To limit the number of API requests**, if the raw comments .jsonl already exists, we allow for scraping only missing videos/performances. **We also provide helpful debugging** in showing the current coverage
and which per-stage performances (as determined from the `official_scores.json`) are missing.

### Step 2 - Analyzing sentiment

Run sentiment analysis over the processed comments, assign each a sentiment score between 0-1, where the backend can be chosen to be:
- `vader` (NTLK's vader), 
- `transformers` (XLM-ROBERTa through HF transformers) 
- `icl_ollama` (custom method using in-context learning on a modern LLM `qwen3:0.6b` through Ollama)

```
python main.py analyze_sentiment --analyzer <vader|transformers|icl_ollama>
```
See `--help` for additional override options (like different config, preprocessed data input, etc.)

Some notes on the sentiment scoring and labels:
- `vader`: the polarity scores from [-1, 1] are mapped to [0, 1]
- `transformers`: we receive a label and probability score from transformer's sentiment analysis pipeline. 
    - If `pos` is contained in the label, we use the probabilty score to map into [0.5, 1.0]
    - If `neg` is contained in the label, we map into [0.0, 0.5]
    - Otherwise, we assume it is neutral (the only other output) and map to 0.5
- `icl_ollama`: we design this method around feeding the below prompt to `qwen3:0.6b` (small model with relatively quick turn-around time i.e. 6 hours on limited GPU hardware <8GB VRAM)
    - We ask the LLM to classify the comment with a label, provide a confidence score (between 0-1) and also a short rationale (for debugging/interpretablity)
    - Given this is a modern LLM and we believe it has the capability of doing this, we introduce an additional label "irrelevant" and instructions to only classify comments directly related to the performer or performance. This is useful as, in this particular competition (Chopin 2025), the results were highly contentious and many negative comments were posted regarding the competition itself and jury (which would have unfairly penalized competitors in our scoring, since such comments would most likely be classified as negative)
    - Qwen3 models have a tendency to spend too much time repeatively thinking - explicitly telling it to not think seems to cut down on the processing time.

Input prompt to `qwen3:0.6b`:

```
    You are a sentiment classifier of YouTube comments from a music competition.
    Only classify comments that seem to be relevant to the performer or performance.
    For example, if the comment is mostly directed to the competition itself or jury, output "irrelevant".
    Do not think.

    Output valid JSON with keys: label (positive/negative/neutral/irrelevant), confidence (0–1), rationale (short).

    TEXT:
    {text}
```

### Step 3 - Generate reports (ranking, correlation, score distributions)

Finally, we can generate reports to interpret our scored comments, aggregate into a single score per performance i.e. video, and compare against the official results.

```
python main.py generate_report
```
If `--sentiment-path` (path to the sentiment-scored .jsonl file) is not provided, we will allow the user to interactively select it based on detected sentiment runs under the given `--results-base` directory ( defaults to `outputs/`):
```
Select a sentiment output:
1) outputs/chopin_2025_icl_ollama/sentiment/comments_scored.jsonl
2) outputs/chopin_2025_transformers/sentiment/comments_scored.jsonl
3) outputs/chopin_2025_vader/sentiment/comments_scored.jsonl
Enter the number of the sentiment file to use: <... user input>
```
Some notes on design choices and the outputs:
- **Aggregating video scores**: to calculate a single score per video, we take a weighted mean of comment sentiment scores, where the weights is the number of likes that each comment received. This is to acocunt for two factors: 
    1) many viewers will usually not post comments themselves, but may like certain comments if it agrees (strongly) with their own opinion - hence, there is some validity to count the sentiment as if that comment was posted as many times as it was liked.
    2) some competitors may receive a much greater following than others (hence much fewer comments, as seen in this competition with 155 total performances)
- Normalizing scores: as mentioned above, some performances/videos have considerably less comments. We found that without applying some kind of normalization, these performances with very few comments 
tend to dominate the top of rankings, since it is much more likely for comments to be generically positive ("Go John Smith!", etc.) and hence benefit from having few negative/positively-nuanced comments.
We support two different styles: 
    - *BM-25 inspired length normalization* - with a symmetric feature, to equally punish too short/long 'documents' from the average 'document' length. Does not perform too well, as it aggressively punishes performances that deviate even slightly from the average number of comments even when tuning `b`.
    - *Bayesian shrinkage* - primarly pulls down the scores of sparsely-commented videos versus the median number of comments, shrinking towards the global mean sentiment. We find this to perform decently well and is set as the default.
- **Rankings**. We compute competitor rankings according to our sentiment-derived scores and output these as `.csv` files (showing our ranking, scores against the official ones, and predicting whether they would advance to the next stage according to the original rank cutoff)
    - Rankings are computed per stage 
    - We also calculate precision @ k, recall, and nDCG for each stage, which are output to a `.json` file
        - k is chosen to be half of the number of competitors that advanced to the next stage, to gauge the accuracy of the top k competitors in our rankings
        - recall is computed over the top-n, where n is exactly the number of competitors that advanced to the next stage (that is why we chose k to be less than n, otherwise precision = recall)
        - nDCG is computed using the official rankings as the ideal ranking. 
        - We define the gain for each position as the number of rounds that this competitor originally advanced (i.e. number of future advancements). Only in the final round, we consider a different gain definition in order to stress the importance of the original awards ranking, following this kind of rule: if 9 competitors placed for final awards, the gain of the first placed competitor is 9, the second is 8, etc.
    - Predicted advancement is defined by whether or not their new ranking meets the same original rank cutoff as the official rankings (i.e. in the Chopin 2025 competition, 40 people advanced to the second round, 20 to the third, and 11 to the finals. We can calculate the original cutoff using the `result` key in the official scoring data)
- **Correlation**. We compute the Pearson correlation coefficients between our sentiment-derived scores and the official jury scores overall, and also plot the correlation scatter plot. This is output to  `.json` and `.png` files respectively.
- **Score Distribution**. We plot the overall distribution of our sentiment-derived scores versus the official jury scores. (If normalized scores are used, we also plot the unnormalized scores for comparison) This is output to a `.png` file.

## Mini-report

We submit all the outputs to this repository. Please view the following directories for the full results of our reporting:
- `outputs/chopin_2025_icl_ollama/`
- `outputs/chopin_2025_transformers/`
- `outputs/chopin_2025_vader/`

A brief analysis of results from running our toolkit on the Chopin 2025 competition:
- VADER technically performed the best in terms of ranking metrics (precision, recall, nDCG), but also awarded the most neutral scores of 0.5 at the comment-level (as it was unable to properly score non-English comments)
- XLM-RoBERTa/transformers's overall (per-video) unnormalized score distribution is remarkably similar to the official jury scores (this likely doesn't really mean anything, as unnormalized scores still unfairly boost performances with very few comments)
- The overall correlation between all three methods against the official jury scores are similar (0.43-0.44), with a weak linear relationship observable in the scatter plots.
    - We expected this to occur, since jury members are human after-all and likely are aware of the general audience sentiment, but of course differ in terms of nuance + professional judgement.
    - (Our audience sentiment gives out high scores much more frequently than the official jury scores - these numbers are mostly accurate as many comments are quite positive with much use of hyperbole)
- `qwen3:0.6b` awarded the fewest neutral scores, which is likely most accurate as it should be quite rare to encounter neutral comments in this context. 
    - However, it also awarded the most positive scores at exactly 1.0 - we note this behaviour seems to be partly due to the small model size. 
    - Previously, we experimented with a large model `qwen3:4b` which preferred to make wider use of the whole 0-1 range, with great nuance in its thinking/rationale. Unfortunately, it was not feasible to actually run it fully, as it would take an estimated 90 hours i.e. 6 days (on limited GPU hardware <8GB VRAM)
    - It was very interesting reading the `rationale` output from this backend, as its reasoning about the comment usually does line up with how I would personally score the comment (although, still with occasional hallucinations)
