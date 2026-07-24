import os, json, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer



SYSTEM_PROMPT = (
    "You are an expert audio scene narrator. "
    "You are given a list of detected sound events with precise timestamps. "
    "Your task is to convert them into natural, accurate English descriptions. "
    "Never mention sounds that are not in the provided detection list. "
    "Always respect the temporal order of events."
)



MAX_TOKENS = {"flat": 50, "temporal": 100, "scene": 100, "cot": 400}



# --------
# HELPER
# --------
# 1.format detections into a string block
def format_detections(detections: pd.DataFrame) -> str:
    """
    detections: DataFrame with columns [event_label, onset, offset, confidence]
    sorted by onset ascending.
    """
    rows = detections.sort_values(by=["onset", "confidence"], ascending=[True, False])
    lines = [
        f"- {row.event_label} ({row.onset:.2f}s – {row.offset:.2f}s, "
        f"confidence: {row.confidence:.2f})"
        for _, row in rows.iterrows()
    ]
    return "\n".join(lines)

# 2. Trim Sentence
def trim_to_last_complete_sentence(text: str) -> str:
    matches = list(re.finditer(r'[.!?](?:\s|$)', text))
    if not matches:
        return text.strip() + "."
    return text[:matches[-1].end()].strip()

# ---------------------------------
# PROMPT BUILDERS — one per format
# ---------------------------------
def build_prompt_flat(detections: pd.DataFrame, clip_duration: float = 10.0) -> str:
    """
    Format 1 — FLAT LIST
    Simple bulleted list. No structural guidance.
    Baseline prompt. Lowest cognitive load on the model.
    """
    det_str = format_detections(detections)
    return (
        f"Audio clip duration: {clip_duration:.1f} seconds.\n\n"
        f"Detected sound events:\n{det_str}\n\n"
        f"Write a single paragraph describing this audio scene in natural English."
    )


def build_prompt_temporal(detections: pd.DataFrame, clip_duration: float = 10.0) -> str:
    """
    Format 2 — TEMPORAL NARRATIVE
    Explicitly instructs the model to follow chronological order.
    Hypothesis: better temporal ordering accuracy (TOA).
    """
    det_str = format_detections(detections)
    return (
        f"Audio clip duration: {clip_duration:.1f} seconds.\n\n"
        f"Detected sound events (in chronological order):\n{det_str}\n\n"
        f"Write a single paragraph that describes the audio scene chronologically. "
        f"Start from the beginning of the clip and narrate events in the order they occur. "
        f"Use temporal connectives such as 'at first', 'then', 'followed by', "
        f"'meanwhile', 'at X seconds', 'by the end'. "
        f"Do not skip any listed event."
    )


def build_prompt_scene(detections: pd.DataFrame, clip_duration: float = 10.0) -> str:
    """
    Format 3 — SCENE STRUCTURED
    Asks the model to decompose into background vs. foreground events first,
    then narrate. Hypothesis: lower hallucination rate (HR) because the model
    reasons about event roles before generating.
    """
    det_str = format_detections(detections)

    # Split detections into long-duration (background) and short-duration (foreground)
    df = detections.copy()
    df["duration"] = df["offset"] - df["onset"]
    median_dur = df["duration"].median()
    bg = df[df["duration"] >= median_dur].sort_values("onset")
    fg = df[df["duration"] <  median_dur].sort_values("onset")

    bg_str = format_detections(bg) if len(bg) > 0 else "  (none)"
    fg_str = format_detections(fg) if len(fg) > 0 else "  (none)"

    return (
        f"Audio clip duration: {clip_duration:.1f} seconds.\n\n"
        f"All detected events:\n{det_str}\n\n"
        f"Background sounds (sustained, long-duration):\n{bg_str}\n\n"
        f"Foreground events (short, prominent):\n{fg_str}\n\n"
        f"Write a single paragraph describing this audio scene. "
        f"Begin with the background soundscape, then describe the foreground events "
        f"as they occur in time. Only describe sounds from the lists above."
    )


def build_prompt_cot(detections: pd.DataFrame, clip_duration: float = 10.0) -> str:
    """
    Format 4 — CHAIN-OF-THOUGHT (CoT)
    Asks the model to reason step-by-step before generating the final description.
    Hypothesis: reduces both temporal errors and hallucinations because the model
    must commit to a structured plan before free-form generation.
    """
    det_str = format_detections(detections)
    return (
        f"Audio clip duration: {clip_duration:.1f} seconds.\n\n"
        f"Detected sound events:\n{det_str}\n\n"
        f"Step 1 — Count and categorize: How many distinct events are there? "
        f"Which overlap in time?\n"
        f"Step 2 — Identify the dominant sound and any transient events.\n"
        f"Step 3 — Write a single-paragraph natural language description of the scene "
        f"that accurately reflects the temporal structure above.\n\n"
        f"Provide Step 1, Step 2 briefly, then Step 3 as the final paragraph."
    )

# -------------------
# INFERENCE FUNCTION
# -------------------
PROMPT_BUILDERS = {
    "flat":     build_prompt_flat,
    "temporal": build_prompt_temporal,
    "scene":    build_prompt_scene,
    "cot":      build_prompt_cot,
}


# batched inference
def generate_narration_batch(
    model_text,
    tokenizer,
    detections_list: list[pd.DataFrame],
    prompt_style: str   = "temporal",
    clip_duration: float = 10.0,
) -> list[str]:
    """
    Args:
        detections_list: list of DataFrames, one per file in the batch
        prompt_style:    same as before — "flat" | "temporal" | "scene" | "cot"
        clip_duration:   applied uniformly across the batch

    Returns:
        list of narration strings, one per input DataFrame
    """
    assert prompt_style in PROMPT_BUILDERS

    # Build one chat-formatted string per sample
    texts = []
    for detections in detections_list:
        user_content = PROMPT_BUILDERS[prompt_style](detections, clip_duration)
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_content},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    # Tokenize all prompts together with left-padding
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,          # guard against very long prompts
    ).to(model_text.device)

    generated_ids = model_text.generate(
        **model_inputs,
        max_new_tokens=MAX_TOKENS[prompt_style],
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Slice off the input prefix from each output
    outputs = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return [
        trim_to_last_complete_sentence(
            tokenizer.decode(out, skip_special_tokens=True).strip()
        )
        for out in outputs
    ]


def format_events_for_text(
    preds_df:   pd.DataFrame,
    filename:   str,
    primary_th: float = 0.10,   # include events above this
    max_events: int   = 8,      # cap to avoid verbose output
) -> list[dict]:
    """
    Returns the top-N high-confidence events for a clip, sorted by onset.
    Falls back to lower-confidence events if the clip has none above primary_th.
    """
    clip_preds = preds_df[preds_df["filename"] == filename].copy()

    # Primary filter
    filtered = clip_preds[clip_preds["confidence"] >= primary_th]

    # Fallback: if completely blank, use top-5 by confidence
    if len(filtered) == 0:
        filtered = clip_preds.nlargest(5, "confidence")

    # Sort by onset, cap to max_events
    filtered = filtered.sort_values(by=["onset", "confidence"], ascending=[True, False]).head(max_events)

    return filtered[["event_label", "onset", "offset", "confidence"]].to_dict("records")


def Save_Text_Generated(model_text, tokenizer, preds_df, style, output_path):
    filenames = preds_df["filename"].unique().tolist()

    with open(output_path, "a") as f:
        batch_detections = [
            pd.DataFrame(format_events_for_text(preds_df, fn))
            for fn in filenames
        ]
        
        generated_texts = generate_narration_batch(
            model_text, tokenizer, batch_detections, prompt_style=style
        )

        # write each row immediately after its batch completes
        for fn, generated_text in zip(filenames, generated_texts):
            row = {"filename": fn, style: generated_text}
            f.write(json.dumps(row) + "\n")
        f.flush()



def Get_Text_Generated(model_text, tokenizer, preds_df, style):
    filenames = preds_df["filename"].unique().tolist()

    batch_detections = [pd.DataFrame(format_events_for_text(preds_df, fn))
                        for fn in filenames]
    
    generated_texts = generate_narration_batch(model_text, tokenizer, batch_detections, prompt_style=style)

    generated_texts_df = pd.DataFrame({'filename': filenames,
                                       'generated_text': generated_texts})
    return generated_texts
    
    
