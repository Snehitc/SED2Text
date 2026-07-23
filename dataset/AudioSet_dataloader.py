import os
import pandas as pd

import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------
# 1. LABEL VOCABULARY
#    Builds a stable class_name → int index mapping from the
#    label MIDs that actually appear in the training TSV.
#    Pass mid_to_name_path to also get human-readable names.
# ----------------------------------------------------------
def build_label_vocab(df: pd.DataFrame, mid_to_name_path: str = None):
    """
    Args:
        df               : the full (ungrouped) train TSV dataframe
        mid_to_name_path : optional path to mid_to_display_name.tsv

    Returns:
        label2idx : dict  {'/m/05zppz': 0, ...}
        idx2label : list  ['/m/05zppz', ...]         (index → MID)
        idx2name  : list  ['Speech', ...]  or None   (index → display name)
    """
    unique_mids = sorted(df["label"].unique().tolist())
    label2idx   = {mid: i for i, mid in enumerate(unique_mids)}
    idx2label   = unique_mids

    idx2name = None
    if mid_to_name_path and os.path.exists(mid_to_name_path):
        mid2name = pd.read_csv(
            mid_to_name_path, sep="\t", header=None, names=["mid", "display_name"]
        ).set_index("mid")["display_name"].to_dict()
        idx2name = [mid2name.get(mid, mid) for mid in idx2label]

    return label2idx, idx2label, idx2name



# --------------
# 2. DATASET
# --------------
class AudioSetStrongDataset(Dataset):
    """
    Each item is ONE 10-second audio clip with ALL its events.

    Returns:
        waveform : Tensor [1, T]   (mono, resampled to target_sr)
        events   : list of dicts  [{"label_idx": int,
                                     "onset": float,   # seconds, clip-relative
                                     "offset": float}, ...]
        meta     : dict           {"segment_id": str, "audio_path": str}
    """

    def __init__(
        self,
        tsv_path:    str,
        audio_dir:   str,
        label2idx:   dict,
        target_sr:   int  = 16000,
        clip_duration: float = 10.0,          # expected clip length in seconds
        audio_ext:   str  = ".flac",
        missing_ok:  bool = False,            # if True, skip missing files silently
    ):
        self.audio_dir     = audio_dir
        self.label2idx     = label2idx
        self.target_sr     = target_sr
        self.clip_duration = clip_duration
        self.audio_ext     = audio_ext
        self.missing_ok    = missing_ok

        # Load & group
        df = pd.read_csv(tsv_path, sep="\t")
        df.columns = df.columns.str.strip()          # guard against whitespace

        # Resolve youtube_id → audio filename 
        df["youtube_id"] = df["segment_id"].str.rsplit("_", n=1).str[0]
        
        if 'eval' in os.path.basename(tsv_path):
            # Define the 9 labels absent from training — remove from eval set
            MISSING_TRAIN_LABELS = {
                '/m/028v0c', '/m/07hvw1', '/t/dd00129', '/m/07p_0gm',
                '/m/01jg1z', '/t/dd00133', '/m/01j3j8', '/t/dd00098', '/m/01jwx6'
            }

            # Before groupby, filter out rows with these labels
            df = df[~df['label'].isin(MISSING_TRAIN_LABELS)].copy()

            # Optionally, also drop clips that become empty after filtering
            # (clips whose only annotation was one of these 9 labels)
            clips_with_events = df.groupby('segment_id').size()
            valid_clips = clips_with_events[clips_with_events > 0].index
            df = df[df['segment_id'].isin(valid_clips)]

            print(f"[AudioSetStrongDataset] Removed annotations for "
                  f"{len(MISSING_TRAIN_LABELS)} unseen label types from eval set. "
                  f"Remaining rows: {len(df)}")

        # Group: one row per clip, events as list-of-dicts
        grouped = (
            df.groupby("segment_id", sort=False)
            .apply(self._group_events, include_groups=False)
            .reset_index()
        )
        grouped.columns = ["segment_id", "events"]
        grouped["youtube_id"] = grouped["segment_id"].str.rsplit("_", n=1).str[0]

        # Filter out clips whose audio file is missing
        grouped["audio_path"] = grouped["youtube_id"].apply(
            lambda ytid: os.path.join(audio_dir, ytid + audio_ext)
        )
        if not missing_ok:
            n_before = len(grouped)
            grouped = grouped[grouped["audio_path"].apply(os.path.exists)].reset_index(drop=True)
            n_after  = len(grouped)
            if n_before != n_after:
                print(f"[AudioSetStrongDataset] Skipped {n_before - n_after} "
                      f"clips with missing audio files.")

        self.samples = grouped  # DataFrame: segment_id | events | audio_path

    # helpers 
    def _group_events(self, sub_df: pd.DataFrame):
        """Called per-group by groupby.apply. Returns list of event dicts."""
        events = []
        for _, row in sub_df.iterrows():
            mid = row["label"]
            if mid not in self.label2idx:
                continue                              # skip labels not in vocab
            events.append({
                "label_idx": self.label2idx[mid],
                "onset":     float(row["start_time_seconds"]),
                "offset":    float(row["end_time_seconds"]),
            })
        return events

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio using soundfile (bypasses torchaudio/torchcodec entirely)."""
        # sf.read returns (numpy_array [T] or [T, C], sample_rate)
        # always_2d=True guarantees shape [T, C] even for mono
        data, sr = sf.read(path, dtype="float32", always_2d=True)

        # [T, C] → [C, T]  (channels-first, matching torchaudio convention)
        waveform = torch.from_numpy(data.T)           # [C, T]

        # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   # [1, T]

        # Resample if needed
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        # Pad / trim to exact clip_duration
        expected_len = int(self.clip_duration * self.target_sr)
        cur_len = waveform.shape[-1]
        if cur_len < expected_len:
            waveform = torch.nn.functional.pad(waveform, (0, expected_len - cur_len))
        else:
            waveform = waveform[:, :expected_len]

        return waveform   # [1, T]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row      = self.samples.iloc[idx]
        waveform = self._load_audio(row["audio_path"])
        events   = row["events"]                     # list of dicts
        meta     = {
            "filename": row["segment_id"],
            "audio_path": row["audio_path"],
        }
        return waveform, events, meta


# ----------------------------------------------------------
# 3. COLLATE FUNCTION
#    Since every clip has a *different number* of events,
#    torch's default collate will fail. We handle it here.
#    Each batch item:
#       waveform : [1, T]
#       events   : variable-length list of dicts
#
#    Output batch:
#       waveforms       : Tensor [B, 1, T]
#       targets         : list[list[dict]]  length B  ← kept as list-of-lists
#                         so DETR's Hungarian matcher can iterate per sample
#       metas           : list[dict]        length B
# ----------------------------------------------------------
def sed_collate_fn(batch):
    waveforms, targets, metas = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)        # [B, 1, T]
    return waveforms, list(targets), list(metas)



# ---------------------
# 4. FACTORY FUNCTION
# ---------------------
def build_audioset_strong_loader(
    tsv_path:     str,
    audio_dir:    str,
    label2idx:    dict,
    batch_size:   int  = 16,
    shuffle:      bool = True,
    num_workers:  int  = 4,
    target_sr:    int  = 16000,
    audio_ext:    str  = ".flac",
    missing_ok:   bool = False,
    pin_memory:   bool = True,
) -> DataLoader:

    dataset = AudioSetStrongDataset(
        tsv_path    = tsv_path,
        audio_dir   = audio_dir,
        label2idx   = label2idx,
        target_sr   = target_sr,
        audio_ext   = audio_ext,
        missing_ok  = missing_ok,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = sed_collate_fn,
        pin_memory  = pin_memory,
        persistent_workers = (num_workers > 0),
    )
    return loader