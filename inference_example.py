import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import torchaudio
import soundfile as sf

from models.PretrainedSED import SED_Prediction, load_pretrained_sed
from models.LanguageModel import Get_Text_Generated
from score import score






# --------- audio load ---------
def load_audio(audio_path, target_sr=16000, clip_duration=10.0):
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)   # [1, T]
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    expected_len = int(clip_duration * target_sr)
    cur_len = waveform.shape[-1]
    
    if cur_len < expected_len:
        waveform = torch.nn.functional.pad(waveform, (0, expected_len - cur_len))
    else:
        waveform = waveform[:, :expected_len]
    meta = [{'filename': os.path.splitext(audio_path.split('/')[-1])[0]}]
    
    return waveform, meta




# --------- Model setup ---------
def model_setup(model_name_SED, model_name_Text, device):
    # 1. PretrainedSED
    model_SED = load_pretrained_sed(model_name_SED, device=device)

    # 2. Language Model
    model_text = AutoModelForCausalLM.from_pretrained(
                    model_name_Text, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_Text)

    # Critical for batched generation with causal LMs:
    # padding must be on the LEFT so all sequences are right-aligned
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model_SED, model_text, tokenizer





if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- models ---
    model_name_SED = "BEATs" # Options: ["BEATs", "ATST-F", "fpasst", "M2D", "ASIT"]
    model_name_Text = "HuggingFaceTB/SmolLM2-360M-Instruct" 
    # Options: ["HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", 
    #           "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"]

    model_SED, model_text, tokenizer = model_setup(model_name_SED, model_name_Text, device)
    model_SED.eval()
    model_text.eval()


    # --- Text style ---
    style = "flat" # Options: [flat, temporal, scene, cot]


    # --- Load audio ---
    audio_path = "sample_files/WsaOJT2SsPg.flac"
    waveform, meta = load_audio(audio_path)

    # --- Prediction ---
    preds_df, gt_df = SED_Prediction(model_SED, waveform, meta)
    generated_text = Get_Text_Generated(model_text, tokenizer, preds_df, style)

    print(f"\nSED Prediction: \n{preds_df.sort_values(by=["onset", "confidence"], ascending=[True, False])}")
    print(f"\nGenerated Text ({style}): {generated_text}")


    # --- Score Prediction ---
    print("\nScore Predicton:")
    df_GeneratedText = pd.DataFrame([[meta[0]['filename'], generated_text[0]]], columns=['filename', style])

    narrations, detections_per_file, filenames = score.fetch_narration_detections(df_GeneratedText, preds_df, style)
    mentions_per_file = score.get_mentions_per_file(narrations)
    mention_embs, label_embs, mention_owner_t, label_owner_t = \
        score.Extract_embeddings(mentions_per_file, detections_per_file, device)
    
    toa, hd, precision, recall, f1, len_mentions, len_labels, ml_ratio = \
        score.Score_computation(filenames, mentions_per_file, detections_per_file, mention_owner_t, label_owner_t, mention_embs, label_embs, narrations)

    score.Score_one_sample(toa, hd, precision, recall, f1, len_mentions, len_labels, ml_ratio)