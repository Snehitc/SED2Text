import os
import yaml
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.AudioSet_dataloader import build_audioset_strong_loader, build_label_vocab
from models.SED import SED_Prediction, load_pretrained_sed
from models.LanguageModel import Save_Text_Generated, Get_Text_Generated

import torch


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


# ------ Create directories to save data ------
def Create_dirs(result_dir, model_name_Text, style):
    SED_Pred_dir = os.path.join(result_dir, 'SED')
    os.makedirs(SED_Pred_dir, exist_ok=True)

    output_dir  = os.path.join(result_dir, 'Text', model_name_Text.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{style}.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)
    return SED_Pred_dir, output_path

@torch.no_grad()
def run_inference(model_SED, model_text, tokenizer, audioset_eval_loader, clip_duration, style):
    model_SED.eval()
    model_text.eval()
    preds_df_list, gt_df_list = [], []
    for waveforms, targets, metas in tqdm(audioset_eval_loader, desc=f"[Generating] SED2Text - {style}"):
        preds_df, gt_df = SED_Prediction(model_SED, waveforms, metas, clip_duration, targets=targets, idx2name=idx2name)
        Save_Text_Generated(model_text, tokenizer, preds_df, style, output_path)
        
        preds_df_list.append(preds_df)
        gt_df_list.append(gt_df)


    all_preds_df = pd.concat(preds_df_list, ignore_index=True)
    all_gt_df = pd.concat(gt_df_list, ignore_index=True)

    all_preds_df.to_csv(os.path.join(SED_Pred_dir, 'all_preds_df.csv'), index=False)
    all_gt_df.to_csv(os.path.join(SED_Pred_dir, 'all_gt_df.csv'), index=False)





if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------ Inputs ------
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Models
    model_name_SED = config['model_name_SED']
    model_name_Text = config['model_name_Text']

    # Text style
    style = config['style']

    # Audio parameters
    sample_rate = config['sample_rate']
    clip_duration = config['clip_duration']


    # Path
    path_meta  = config['path_meta']
    path_audio = config['path_audio']
    train_file = config['train_file']
    eval_file = config['eval_file']
    mid2name_file = config['mid2name_file']
    mid2name = os.path.join(path_meta, mid2name_file)
    result_dir = config['result_dir']

    # Parameters
    batch_size = config['batch_size']
    num_workers = config['num_workers']


    # --------- Dataloader ---------
    df_train = pd.read_csv(os.path.join(path_meta, train_file), sep="\t")
    label2idx, idx2label, idx2name = build_label_vocab(df_train, mid2name)

    audioset_eval_loader = build_audioset_strong_loader(
                                tsv_path   = os.path.join(path_meta, eval_file),
                                audio_dir  = path_audio,
                                target_sr = sample_rate,
                                label2idx  = label2idx,
                                batch_size = batch_size,
                                shuffle    = False,
                                num_workers = num_workers,)

    # ------ Setup and run inference ------
    SED_Pred_dir, output_path = Create_dirs(result_dir, model_name_Text, style)
    model_SED, model_text, tokenizer = model_setup(model_name_SED, model_name_Text, device)
    run_inference(model_SED, model_text, tokenizer, audioset_eval_loader, clip_duration, style)
