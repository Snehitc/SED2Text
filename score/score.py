import os
import re
import torch
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import torch
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util
import yaml
from tabulate import tabulate




GENERIC_MENTIONS = {
    "something", "someone", "anything", "everything",
    "thing", "things", "audio", "clip", "event",
    "we", "they", "it", "this", "that", "there"
}

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


def build_label_reference(gt_events: pd.DataFrame) -> str:
    rows = gt_events.sort_values("onset")
    parts = [
        f"{row.event_label} from {row.onset:.1f}s to {row.offset:.1f}s"
        for _, row in rows.iterrows()
    ]
    return ". ".join(parts) + "."


# ----------------------------------------------------------------------
# Stage 0: Pre-fetch narrations and detections per file
# ----------------------------------------------------------------------
def fetch_narration_detections(df_GeneratedText, all_preds_df, style):
    narrations = []
    detections_per_file = []

    df_indexed = df_GeneratedText.set_index("filename")
    filenames = all_preds_df['filename'].unique().tolist()
    for fn in tqdm(filenames, desc="Loading narrations & detections", unit="file"):
        narrations.append(df_indexed.loc[fn, style])
        events = format_events_for_text(all_preds_df, fn)
        detections_per_file.append(pd.DataFrame(events))

    return narrations, detections_per_file, filenames



# ----------------------------------------------------------------------
# Preprocessing: concept-level mentions only, no raw n-gram soup
# ----------------------------------------------------------------------
def normalize_mention(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'["“”]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_event_mentions(doc, max_len: int = 4) -> list[str]:
    mentions = []
    for c in doc.noun_chunks:
        txt = normalize_mention(c.text)
        if 0 < len(txt.split()) <= max_len and txt not in GENERIC_MENTIONS:
            mentions.append(txt)
    for tok in doc:
        if tok.pos_ == "VERB" and tok.lemma_.lower() not in GENERIC_MENTIONS:
            mentions.append(tok.lemma_.lower())
    # NOTE: n-gram window loop removed — this was the source of inflation
    return list(dict.fromkeys(mentions))

def get_mentions_per_file(narrations):
    mentions_per_file = []
    nlp = spacy.load("en_core_web_sm")
    docs = nlp.pipe(narrations, n_process=4, batch_size=200)

    for doc in tqdm(docs, total=len(narrations), desc="Extracting event mentions", unit="file"):
        mentions_per_file.append(extract_event_mentions(doc, max_len=4))
    return mentions_per_file


def Extract_embeddings(mentions_per_file, detections_per_file, device):
    model_ST = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    flat_mentions, mention_owner = [], []
    for i, mlist in enumerate(mentions_per_file):
        flat_mentions.extend(mlist)
        mention_owner.extend([i] * len(mlist))

    flat_labels, label_owner = [], []
    for i, det in enumerate(detections_per_file):
        labs = det["event_label"].astype(str).str.lower().str.strip().tolist() if len(det) else []
        labs = list(dict.fromkeys(labs))
        flat_labels.extend(labs)
        label_owner.extend([i] * len(labs))

    print(f"Encoding {len(flat_mentions)} mentions and {len(flat_labels)} labels on GPU...")

    mention_embs = model_ST.encode(
        flat_mentions, batch_size=256, convert_to_tensor=True,
        show_progress_bar=True, device=model_ST.device
    )
    label_embs = model_ST.encode(
        flat_labels, batch_size=256, convert_to_tensor=True,
        show_progress_bar=True, device=model_ST.device
    )

    mention_owner_t = torch.tensor(mention_owner)
    label_owner_t = torch.tensor(label_owner)

    return mention_embs, label_embs, mention_owner_t, label_owner_t





# ----------------------------------------------------------------------
# Stage C: Score Computation Functions
# ----------------------------------------------------------------------
def temporal_ordering_accuracy(
    mentions: list[str],
    mention_embs_i,          # [M, D] tensor for THIS file only
    detections: pd.DataFrame,
    label_embs_i,             # [L, D] tensor for THIS file only (built from UNIQUE labels)
    sim_threshold: float = 0.6,
) -> float:
    sorted_dets = detections.sort_values(by=["onset", "confidence"], ascending=[True, False])

    # Deduplicate while preserving first-occurrence (i.e. earliest onset) order —
    # this MUST match how flat_labels was built in Stage B.
    labels_in_order = list(dict.fromkeys(
        sorted_dets["event_label"].str.lower().str.strip().tolist()
    ))

    if len(mentions) == 0 or len(labels_in_order) == 0:
        return 1.0

    # Safety check: label_embs_i must have exactly len(labels_in_order) rows
    assert label_embs_i.shape[0] == len(labels_in_order), (
        f"Mismatch: label_embs_i has {label_embs_i.shape[0]} rows, "
        f"but labels_in_order has {len(labels_in_order)} unique labels."
    )

    sims = util.cos_sim(label_embs_i, mention_embs_i).cpu().numpy()  # [L, M]
    best_mention_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    positions = {}
    for j, label in enumerate(labels_in_order):
        if best_sim[j] >= sim_threshold:
            positions[label] = best_mention_idx[j]

    found_labels = [l for l in labels_in_order if l in positions]
    if len(found_labels) < 2:
        return 1.0

    correct = sum(
        positions[found_labels[i]] < positions[found_labels[i + 1]]
        for i in range(len(found_labels) - 1)
    )
    return correct / (len(found_labels) - 1)


def compute_grounding_scores(mention_embs_i, label_embs_i, sim_threshold=0.6):
    """
    mention_embs_i: [M, D] tensor for one file's deduplicated mentions
    label_embs_i:   [L, D] tensor for one file's unique detected labels
    Returns precision, recall (=EC), f1 via optimal one-to-one assignment.
    """
    M, L = mention_embs_i.shape[0], label_embs_i.shape[0]
    if M == 0 and L == 0:
        return 1.0, 1.0, 1.0
    if M == 0:
        return 1.0, 0.0, 0.0
    if L == 0:
        return 0.0, 1.0, 0.0

    sims = util.cos_sim(mention_embs_i, label_embs_i).cpu().numpy()  # [M, L]
    cost = -sims  # maximize similarity -> minimize negative
    row_idx, col_idx = linear_sum_assignment(cost)

    matched = sims[row_idx, col_idx] >= sim_threshold
    tp = matched.sum()

    precision = tp / M          # fraction of mentions that are grounded (1 - HR)
    recall    = tp / L          # this is EC
    f1        = 0.0 if (precision + recall) == 0 else 2*precision*recall/(precision+recall)
    return precision, recall, f1


def compute_hallucination_density(mentions, mention_embs_i, label_embs_i, word_count, sim_threshold=0.6):
    """
    Hallucination Density: fraction of narration WORDS (not mention-chunks)
    that come from ungrounded mentions. Normalizes against text length so
    verbosity no longer drives the score up mechanically.
    """
    M, L = mention_embs_i.shape[0], label_embs_i.shape[0]
    if M == 0:
        return 0.0
    if L == 0:
        ungrounded_word_count = sum(len(m.split()) for m in mentions)
        return ungrounded_word_count / max(word_count, 1)

    sims = util.cos_sim(mention_embs_i, label_embs_i).cpu().numpy()  # [M, L]
    ungrounded_mask = sims.max(axis=1) < sim_threshold
    ungrounded_word_count = sum(
        len(m.split()) for m, flag in zip(mentions, ungrounded_mask) if flag
    )
    return ungrounded_word_count / max(word_count, 1)

def compute_va_f1(f1: float, mention_count: int) -> float:
    """
    Verbosity-Adjusted F1: dampens F1 by log(1 + mention_count),
    penalizing narrations that reach high F1 through sheer mention volume
    rather than concise, grounded content.
    """
    return f1 / np.log1p(mention_count)


# ----------------------------------------------------------------------
# Stage Score Computation over mentions and labels
# ----------------------------------------------------------------------
def Score_computation(filenames, mentions_per_file, detections_per_file, mention_owner_t, label_owner_t, mention_embs, label_embs, narrations):
    temporal_ordering_list, hd_list, precision_list, recall_list, f1_list = [], [], [], [], []
    len_mentions, len_labels = [], []


    for i in tqdm(range(len(filenames)), desc="Computing Scores", unit="file"):
        m_idx = (mention_owner_t == i).nonzero(as_tuple=True)[0]
        l_idx = (label_owner_t == i).nonzero(as_tuple=True)[0]

        mention_embs_i = mention_embs[m_idx]
        label_embs_i   = label_embs[l_idx]

        p, r, f1 = compute_grounding_scores(mention_embs_i, label_embs_i)
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)

        word_count = len(narrations[i].split())
        hd_list.append(compute_hallucination_density(mentions_per_file[i], mention_embs_i, label_embs_i, word_count))

        len_mentions.append(len(mentions_per_file[i]))
        len_labels.append(label_embs_i.shape[0])


        toa = temporal_ordering_accuracy(
            mentions_per_file[i],
            mention_embs_i,
            detections_per_file[i],
            label_embs_i,
        )
        temporal_ordering_list.append(toa)


        mention_count_i = len(mentions_per_file[i])

    ml_ratio_list = [m / l if l > 0 else np.nan
                    for m, l in zip(len_mentions, len_labels)]
    

    return temporal_ordering_list, hd_list, precision_list, recall_list, f1_list, len_mentions, len_labels, ml_ratio_list




# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
def Score_Save_display(score_dir, filenames, temporal_ordering_list, hd_list, precision_list, recall_list, f1_list, len_mentions, len_labels, ml_ratio_list):
    # Save Score Data
    score_data = {'filename': filenames,
                'ToA': temporal_ordering_list,
                'HD': hd_list,
                'Precision': precision_list,
                'Recall': recall_list,
                'F1': f1_list,
                'M_mentions': len_mentions,
                'L_labels': len_labels,
                'ML_ratio': ml_ratio_list,}
    df_score = pd.DataFrame(score_data)
    score_filename = os.path.join(score_dir, f"scores_{style}.csv")
    if os.path.exists(score_filename):
        os.remove(score_filename)
    df_score.to_csv(score_filename, index=False)

    # Average values of Score to Display
    avg_TOA = np.mean(temporal_ordering_list)
    avg_HD = np.mean(hd_list)
    avg_P = np.mean(precision_list)
    avg_R = np.mean(recall_list)
    avg_F1 = np.mean(f1_list)
    avg_M = np.mean(len_mentions)
    avg_L = np.mean(len_labels)  
    avg_ML_ratio = np.nanmean(ml_ratio_list)

    Table = [[avg_TOA, avg_HD, avg_P, avg_R, avg_F1, avg_M, avg_L, avg_ML_ratio]]
    headers = ['TOA', 'HD', 'Precision', 'Recall', 'F1', 'avg M', 'avg L', 'M/L ratio']
    print(f"\nFile Path: {score_filename}")
    print(tabulate(Table, headers=headers, tablefmt="grid", floatfmt=".3f"))


def Score_one_sample(toa, hd, precision, recall, f1, len_mentions, len_labels, ml_ratio):
    Table = [[toa[0], hd[0], precision[0], recall[0], f1[0], len_mentions[0], len_labels[0], ml_ratio[0]]]
    headers = ['TOA', 'HD', 'Precision', 'Recall', 'F1', 'avg M', 'avg L', 'M/L ratio']
    print(tabulate(Table, headers=headers, tablefmt="grid", floatfmt=".3f"))



if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    model_name_SED = config['model_name_SED']
    model_name_Text = config['model_name_Text']
    style = config['style']
    result_dir = os.path.join(config['result_dir'])
    output_dir  = os.path.join(result_dir, 'Text', model_name_Text.split('/')[-1])
    output_path = os.path.join(output_dir, f"{style}.jsonl")
    score_dir  = os.path.join(result_dir, 'Score', model_name_Text.split('/')[-1])
    os.makedirs(score_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_gt_df = pd.read_csv(os.path.join(result_dir, "SED", "all_gt_df.csv"))
    all_preds_df = pd.read_csv(os.path.join(result_dir, "SED", "all_preds_df.csv"))
    
    df_GeneratedText = pd.read_json(output_path, lines=True)
    

    narrations, detections_per_file, filenames = fetch_narration_detections(df_GeneratedText, all_preds_df, style)

    mentions_per_file = get_mentions_per_file(narrations)
    mention_embs, label_embs, mention_owner_t, label_owner_t = \
        Extract_embeddings(mentions_per_file, detections_per_file, device)
    
    temporal_ordering_list, hd_list, precision_list, recall_list, f1_list, len_mentions, len_labels, ml_ratio_list = \
        Score_computation(filenames, mentions_per_file, detections_per_file, mention_owner_t, label_owner_t, mention_embs, label_embs, narrations)

    Score_Save_display(score_dir, filenames, temporal_ordering_list, 
                       hd_list, precision_list, recall_list, f1_list, 
                       len_mentions, len_labels, ml_ratio_list)