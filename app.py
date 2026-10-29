import json

import gradio as gr
import pandas as pd
from mlx_lm import generate, load

from part2_gloss import sb_sentence_to_gloss
from prompt import INSTRUCTIONS

MODELS = {
    "Gemma 3 4B (fine-tuned)": ("mlx-community/gemma-3-4b-it-4bit", "adapters/gemma3-4b"),
    "Llama 3.2 3B (fine-tuned)": ("mlx-community/Llama-3.2-3B-Instruct-4bit", "adapters/llama32-3b"),
}

_cache = {}


def get_model(name):
    if name not in _cache:
        base, adapter = MODELS[name]
        _cache[name] = load(base, adapter_path=adapter)
    return _cache[name]


def analyze(sentence, model_name):
    model, tokenizer = get_model(model_name)
    prompt_text = INSTRUCTIONS + sentence.strip()
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=400, verbose=False)

    result = {"sentence": sentence, "raw_output": raw}
    try:
        parsed = json.loads(raw)
        result["sb_words"] = parsed.get("sb_words", "")
        result["sb_sentence"] = parsed.get("sb_sentence", "")
        try:
            result["part2_gloss"] = sb_sentence_to_gloss(result["sb_sentence"])
        except Exception as e:
            result["part2_gloss"] = ""
            result["gloss_error"] = str(e)
    except Exception as e:
        result["parse_error"] = str(e)
    return result


def chat_fn(message, history, model_name):
    result = analyze(message, model_name)
    if "parse_error" in result:
        return (
            f"Model output wasn't valid JSON ({result['parse_error']}).\n\n"
            f"Raw output:\n```\n{result['raw_output']}\n```"
        )
    parts = [
        f"**SB of the word:**\n```\n{result['sb_words']}\n```",
        f"**SB of the sentence:**\n```\n{result['sb_sentence']}\n```",
    ]
    if result.get("part2_gloss"):
        parts.append(f"**Traditional gloss (Part 2):**\n{result['part2_gloss']}")
    elif result.get("gloss_error"):
        parts.append(f"_Part 2 gloss not available for this structure: {result['gloss_error']}_")
    return "\n\n".join(parts)


def process_csv(file, model_name, progress=gr.Progress()):
    if file is None:
        return None, None
    df = pd.read_csv(file.name)

    sentence_col = None
    for c in df.columns:
        if c.strip().lower() in ("sentence", "sentences"):
            sentence_col = c
            break
    if sentence_col is None:
        sentence_col = df.columns[0]

    sb_words_out, sb_sentence_out, gloss_out, error_out = [], [], [], []
    sentences = df[sentence_col].astype(str).tolist()
    for sentence in progress.tqdm(sentences, desc="Analyzing sentences"):
        result = analyze(sentence, model_name)
        sb_words_out.append(result.get("sb_words", ""))
        sb_sentence_out.append(result.get("sb_sentence", ""))
        gloss_out.append(result.get("part2_gloss", ""))
        error_out.append(result.get("parse_error", ""))

    df["predicted_sb_words"] = sb_words_out
    df["predicted_sb_sentence"] = sb_sentence_out
    df["predicted_part2_gloss"] = gloss_out
    df["parse_error"] = error_out

    out_path = "predictions_output.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df, out_path


with gr.Blocks(title="Shabdabodha Predictor") as demo:
    gr.Markdown("# Shabdabodha Predictor\nRun your fine-tuned Gemma / Llama adapters on Sanskrit sentences.")
    model_dropdown = gr.Dropdown(
        choices=list(MODELS.keys()), value=list(MODELS.keys())[0], label="Model"
    )

    with gr.Tab("Chat (single sentence)"):
        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[model_dropdown],
            examples=[["बालः पादाभ्यां गृहं गच्छति।"], ["रामः पचति।"]],
        )

    with gr.Tab("CSV batch"):
        gr.Markdown("Upload a CSV with a `Sentence` column (or sentences in the first column).")
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        run_btn = gr.Button("Run predictions", variant="primary")
        output_table = gr.Dataframe(label="Results", wrap=True)
        download_file = gr.File(label="Download results CSV")
        run_btn.click(
            process_csv,
            inputs=[file_input, model_dropdown],
            outputs=[output_table, download_file],
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
