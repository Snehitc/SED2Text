import sys
sys.path.append('/home/u5049807/PretrainedSED/')

import os
import torch
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter as scipy_median_filter

# ----------------------------------------
# PretrainedSED imports (from their repo)
# ----------------------------------------
from PretrainedSED.helpers.decode import batched_decode_preds
from PretrainedSED.helpers.encode import ManyHotEncoder
from PretrainedSED.models.prediction_wrapper import PredictionsWrapper
from PretrainedSED.data_util import audioset_classes


# ----------
# CONSTANTS
# ----------
SAMPLE_RATE     = 16_000
CLIP_DURATION   = 10.0          # seconds — fixed for AudioSet-Strong
CLIP_SAMPLES    = int(CLIP_DURATION * SAMPLE_RATE)
PRETRAINED_CLASSES = audioset_classes.as_strong_train_classes  # list[str], 447 items
N_CLASSES       = len(PRETRAINED_CLASSES)                       # 447
MEDIAN_FILTER_FRAMES = 12   # 12 frames × 40ms = 0.48s, matches paper's postprocessing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------
# MODEL LOADER
# -------------
def load_pretrained_sed(model_name: str = "BEATs", device: str = "cuda") -> PredictionsWrapper:
    """
    Loads a PretrainedSED model by name.
    Supported: "BEATs", "ATST-F", "fpasst", "M2D", "ASIT""
    """
    
    from PretrainedSED.models.beats.BEATs_wrapper import BEATsWrapper
    from PretrainedSED.models.atstframe.ATSTF_wrapper import ATSTWrapper
    from PretrainedSED.models.frame_passt.fpasst_wrapper import FPaSSTWrapper
    from PretrainedSED.models.m2d.M2D_wrapper import M2DWrapper
    from PretrainedSED.models.asit.ASIT_wrapper import ASiTWrapper

    if model_name == "BEATs":
        backbone = BEATsWrapper()
        model = PredictionsWrapper(backbone, checkpoint="BEATs_strong_1")
    elif model_name == "ATST-F":
        backbone = ATSTWrapper()
        model = PredictionsWrapper(backbone, checkpoint="ATST-F_strong_1")
    elif model_name == "fpasst":
        backbone = FPaSSTWrapper()
        model = PredictionsWrapper(backbone, checkpoint="fpasst_strong_1")
    elif model_name == "M2D":
        backbone = M2DWrapper()
        model = PredictionsWrapper(backbone, embed_dim=3840, checkpoint="M2D_strong_1")
    elif model_name == "ASIT":
        backbone = ASiTWrapper()
        model = PredictionsWrapper(backbone, checkpoint="ASIT_strong_1")
    else:
        raise NotImplementedError(f"Model '{model_name}' not supported here.")

    model.eval()
    model.to(device)
    print(f"[PretrainedSED] Loaded '{model_name}' → device={device}")
    return model


# -----------------------------------------------------------
# SINGLE-BATCH INFERENCE
# Returns a raw sigmoid probability tensor [B, C=447, T=250]
# -----------------------------------------------------------
@torch.no_grad()
def run_inference_batch(
    model:    PredictionsWrapper,
    waveform: torch.Tensor,          # [B, 1, T_samples] or [B, T_samples]
    device:   str = "cuda",
) -> torch.Tensor:
    """
    Runs mel + forward pass, returns sigmoid probabilities.

    Output: [B, C=447, T=250]  float32, values in [0, 1]
    """
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)          # [B, T_samples]
    waveform = waveform.float().to(device)

    mel      = model.mel_forward(waveform)      # [B, 1, Mel, T]
    y_strong, _ = model(mel)                    # [B, C=447, T=250]  raw logits
    return torch.sigmoid(y_strong).float()      # [B, C=447, T=250]



@torch.no_grad()
def _attach_confidence(
    preds_batch: pd.DataFrame,
    probs_filtered: torch.Tensor,   # [B, C=447, T=250]
    segment_ids: list,
    clip_duration: float = 10.0,
) -> pd.DataFrame:
    """
    Attaches mean probability as confidence to each detected event row.
    probs_filtered: sigmoid probabilities after median filter, CPU tensor.
    """
    if len(preds_batch) == 0:
        preds_batch["confidence"] = pd.Series(dtype=float)
        return preds_batch

    n_frames = probs_filtered.shape[2]               # 250
    frames_per_sec = n_frames / clip_duration         # 25.0

    # Build fast lookup: segment_id → batch index
    seg2idx = {seg: i for i, seg in enumerate(segment_ids)}

    # Build fast lookup: label → class index
    label2cls = {lbl: i for i, lbl in enumerate(PRETRAINED_CLASSES)}

    confidences = []
    for _, row in preds_batch.iterrows():
        # Strip .wav suffix for lookup — segment_ids don't have it
        fn = str(row["filename"]).replace(".wav", "")
        b_idx   = seg2idx.get(fn, -1)
        cls_idx = label2cls.get(row["event_label"], -1)

        if b_idx == -1 or cls_idx == -1:
            confidences.append(0.1)
            continue

        f_start = max(0, int(row["onset"]  * frames_per_sec))
        f_end   = min(n_frames, int(row["offset"] * frames_per_sec) + 1)

        if f_start >= f_end:
            f_end = f_start + 1

        conf = round(float(probs_filtered[b_idx, cls_idx, f_start:f_end].mean()), 2)
        confidences.append(conf)

    preds_batch = preds_batch.copy()
    preds_batch["confidence"] = confidences
    return preds_batch


# ------------------------------------------------------------------
# DECODE PREDICTIONS using PretrainedSED's own batched_decode_preds
# ------------------------------------------------------------------
def decode_predictions_pretrained(
    probs:        torch.Tensor,    # [B, C=447, T=250]  CPU tensor, float32
    segment_ids:  list,            # list[str], length B
    threshold:    float = 0.1,
    median_filter_frames: int = MEDIAN_FILTER_FRAMES,
    clip_duration: float = CLIP_DURATION,
) -> pd.DataFrame:
    """
    Decodes frame-level probabilities into event segments using
    PretrainedSED's ManyHotEncoder + batched_decode_preds.

    Returns:
        DataFrame with columns: [filename, event_label, onset, offset, confidence]
        'confidence' = mean probability of the detection segment
    """
    # ManyHotEncoder requires audio_len — constant 10.0s for AudioSet-Strong
    encoder = ManyHotEncoder(
        PRETRAINED_CLASSES,
        audio_len=clip_duration,
    )

    # batched_decode_preds expects:
    #   strong_preds : Tensor [B, C, T]  (already sigmoid-applied)
    #   filenames    : list[str]         (used as the 'filename' column)
    #   encoder      : ManyHotEncoder
    #   median_filter: int               (number of frames)
    #   thresholds   : tuple or list of floats
    (
        scores_unprocessed,       # dict[th → DataFrame]  before median filter
        scores_postprocessed,     # dict[th → DataFrame]  after median filter
        decoded_predictions,      # dict[th → DataFrame]  thresholded events
    ) = batched_decode_preds(
        probs,
        segment_ids,
        encoder,
        median_filter = median_filter_frames,
        thresholds    = (threshold,),   # single threshold
    )

    preds_df = decoded_predictions[threshold].copy()

    # Rename 'event_label' already correct; add confidence from postprocessed scores
    # batched_decode_preds output columns: event_label, onset, offset, filename
    # We add confidence from the raw probability at the midpoint of the segment
    if "confidence" not in preds_df.columns:
        preds_df["confidence"] = threshold   # fallback: constant confidence

    return preds_df.reset_index(drop=True)


def SED_Prediction(model_SED, waveforms, metas, targets=None, idx2name=None):
    segment_ids = [os.path.splitext(m["filename"])[0] for m in metas]

    # Inference
    probs = run_inference_batch(model_SED, waveforms, device)   # [B, C, T]

    # Apply median filter per class (PSDS needs filtered probs)
    # probs_cpu: [B, C, T] → filter along T axis
    probs_filtered = torch.from_numpy(
        scipy_median_filter(probs.cpu().numpy(), size=(1, 1, MEDIAN_FILTER_FRAMES))
    )                                                        # [B, C, T]
    # Decode predictions
    preds_df = decode_predictions_pretrained(
        probs.cpu(),
        segment_ids,
        threshold            = 0.1,
        median_filter_frames = MEDIAN_FILTER_FRAMES,
        clip_duration        = CLIP_DURATION,
    )

    preds_df = _attach_confidence(
        preds_df, probs_filtered, segment_ids, CLIP_DURATION)

    preds_df = preds_df if not preds_df.empty else pd.DataFrame(
        columns=["filename", "event_label", "onset", "offset", "confidence"])
    preds_df['filename'] = preds_df['filename'].apply(lambda x: os.path.splitext(x)[0])

    # Build GT from DataLoader targets
    gt_rows = []
    if targets:
        for b, events in enumerate(targets):
            seg_id = segment_ids[b]
            for ev in events:
                label  = idx2name[ev["label_idx"]]
                onset  = round(float(ev["onset"]),  3)
                offset = round(float(ev["offset"]), 3)
                if offset <= onset:
                    continue
                gt_rows.append({
                    "filename":    seg_id,
                    "event_label": label,
                    "onset":       onset,
                    "offset":      offset,
                })

    gt_df = pd.DataFrame(gt_rows) if gt_rows else pd.DataFrame(
            columns=["filename", "event_label", "onset", "offset"])
            
    return preds_df, gt_df