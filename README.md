# SHABDABODHA-LLM 2.0

Code for the paper *SHABDABODHA-LLM 2.0: Computational Sanskrit Śābdabodha: A Vyākaraṇa Approach*
(Tapas Khanra, Radhe Shyam Salopanthula, Malhar Kulkarni).

The project generates word-level (WM) and sentence-level (SM) *śābdabodha* (verbal cognition) representations for
Sanskrit sentences with fine-tuned language models, scores the output against human-verified reference annotations, and
includes a small local app for trying sentences and CSV files.

## Contents

| File | Purpose |
|---|---|
| `prompt.py` | The task instruction used for training and inference |
| `prepare_data.py` | Converts the released dataset into MLX-LM chat data |
| `app.py` | Local Gradio app: chat window and CSV batch prediction |
| `sb_metrics.py` | Automatic measures: well-formedness, exact match, outer label, head, kāraka-label F1 |
| `score_heldout.py` | Reproduces the held-out (20 sentence) table |
| `human_eval_stats.py` | Reproduces the human-evaluation means, agreement and significance tests |
| `part2_gloss.py` | Draft rule-based converter from the sentence-level expression to the traditional one-line gloss |

## Data and models

| Resource | Link |
|---|---|
| Dataset (sentences, annotations, human scores, model outputs) | https://huggingface.co/datasets/samskritarnava-iitb/shabdabodha-dataset |
| Gemma 3 4B LoRA adapter | https://huggingface.co/samskritarnava-iitb/gemma-3-4b-it-shabdabodha-lora |
| Llama 3.2 3B LoRA adapter | https://huggingface.co/samskritarnava-iitb/Llama-3.2-3B-Instruct-shabdabodha-lora |
| Hugging Face organization | https://huggingface.co/samskritarnava-iitb |

```bash
hf download samskritarnava-iitb/shabdabodha-dataset --repo-type dataset --local-dir shabdabodha-dataset
hf download samskritarnava-iitb/gemma-3-4b-it-shabdabodha-lora --local-dir adapters/gemma3-4b
hf download samskritarnava-iitb/Llama-3.2-3B-Instruct-shabdabodha-lora --local-dir adapters/llama32-3b
```

The dataset and model cards on Hugging Face link back to this repository.

### Dataset at a glance

| Config | Content | Rows |
|---|---|---|
| `sentences` | Sentences with word-level and sentence-level annotations (`train` / `validation` / `test`) | 224 (183 / 21 / 20) |
| `human_eval_outputs` | The 198 human-evaluated sentences with the reference and four model outputs | 198 |
| `human_eval_scores` | Word-level and sentence-level scores (1-10) from three annotators, long format | 2970 |
| `heldout_predictions` | Outputs of four systems on the 20 test sentences | 20 |

```python
from datasets import load_dataset
sentences = load_dataset("samskritarnava-iitb/shabdabodha-dataset", "sentences")
```

The 198 human-evaluated sentences are train/validation sentences, not test sentences; the held-out evaluation uses the 20 test sentences.

## Setup

MLX-LM needs Apple Silicon (macOS). The scoring scripts run on any machine.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
# 1. training data (identical to the data used for the reported runs)
python prepare_data.py --data shabdabodha-dataset

# 2. LoRA fine-tuning (about 35 min per model on an Apple M4, 16 GB)
python -m mlx_lm lora --model mlx-community/gemma-3-4b-it-4bit --train --data data/mlx \
  --adapter-path adapters/gemma3-4b --iters 300 --batch-size 2 --num-layers 16 --mask-prompt \
  --steps-per-report 20 --steps-per-eval 50 --save-every 100 --seed 42
python -m mlx_lm lora --model mlx-community/Llama-3.2-3B-Instruct-4bit --train --data data/mlx \
  --adapter-path adapters/llama32-3b --iters 300 --batch-size 2 --num-layers 16 --mask-prompt \
  --steps-per-report 20 --steps-per-eval 50 --save-every 100 --seed 42

# 3. numbers reported in the paper
python score_heldout.py --data shabdabodha-dataset
python human_eval_stats.py --data shabdabodha-dataset

# 4. local app at http://localhost:7860
python app.py
```

If loading hangs on a network check, set `HF_HUB_OFFLINE=1` and `GRADIO_ANALYTICS_ENABLED=False`.

## Results (preliminary)

- Fine-tuned Gemma 3 4B and Llama 3.2 3B reproduce the output notation but not yet its finer kāraka distinctions
  (for example गौणकर्म vs मुख्यकर्म), which is consistent with the small and skewed corpus (224 sentences).
- Human evaluation (198 entries, 3 annotators, 10-point scale): the four fine-tuned systems score 5.4 to 5.6 against 9.0 for the
  reference annotations, with no model clearly ahead.
- Annotators A1 and A2 give near-identical scores, so their ratings should not be treated as independent.

See the paper for details and caveats.

## Known issues

These are also documented in the dataset card.

- Annotations are released verbatim. Notation irregularities (alternative separators, stray hyphens, non-standard person markers,
  finite verb forms as heads) are flagged per sentence in `notation_flags`: 78 of 224 sentences carry at least one flag.
- In some references the verbal noun differs between the word-level and sentence-level annotation (e.g. पचन vs पाक for रामः पचति।).
- Four human-score cells were invalid in the source sheet (a score of 99, or blank, all in the reference word-level column); they
  are left empty with a `note`. One sentence appears twice among the 198 entries.
- Annotators A1 and A2 give identical scores on 92% of cells (alpha 0.97) while each agrees with A3 at alpha of about 0.14, so
  A1 and A2 should not be treated as independent.
- The held-out test set has only 20 sentences, so differences of a few sentences are not meaningful.

## Part-2 gloss (draft)

`part2_gloss.py` turns a sentence-level expression into the traditional one-line gloss, for example
`वर्तमानकालिकः बाल-अभिन्न-प्रथमपुरुषैकवचन-कर्तृकः पादकरणकः गृहकर्मकः गमनानुकूलव्यापारः`. It is correct on simple sentences and has not
been validated on the whole dataset; many constructions (vocative, negation, operator-wrapped sentences) are not handled.

## License

Code: MIT. Dataset: CC BY 4.0. The adapters are derivatives of Gemma 3 and Llama 3.2 and are subject to their licenses.

## Citation

```bibtex
@misc{shabdabodha_llm_2026,
  title  = {SHABDABODHA-LLM 2.0: Computational Sanskrit Śābdabodha: A Vyākaraṇa Approach},
  author = {Khanra, Tapas and Salopanthula, Radhe Shyam and Kulkarni, Malhar},
  year   = {2026},
  note   = {Code}
}
```
