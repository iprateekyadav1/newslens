"""
NewsLens — Dataset

Provides two data sources (merged at training time):

1. SYNTHETIC — ~400 programmatically generated examples covering all
   five bias categories and three political lean classes.
   Deterministic (seeded) so results are reproducible.
   These are not perfect but give the model a strong prior signal.

2. MBIC LOADER — optional; loads the real Media Bias In Context dataset
   (Spinde et al., 2021) if you place `mbic.csv` in data/raw/.
   Download: https://github.com/Media-Bias-Group/Media-Bias-Identification

Label format per example:
  text            : str
  bias_labels     : list[int] length 5, multi-hot (one per BIAS_CATEGORIES)
  lean_label      : int  0=left  1=center  2=right
  bias_intensity  : float [0,1]
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from config import BIAS_CATEGORIES, LEAN_LABELS, DATA_DIR, MODEL_MAX_LENGTH

# ── Label indices ─────────────────────────────────────────────────────────────
LL = 0   # Loaded Language
FR = 1   # Framing
EP = 2   # Epistemic Manipulation
AN = 3   # Anchoring
SE = 4   # Sensationalism
FB = 5   # False Balance       (new)
WA = 6   # Whataboutism        (new)
IG = 7   # In-Group Framing    (new)

NUM_BIAS = len(BIAS_CATEGORIES)   # 8
LEFT, CENTER, RIGHT = 0, 1, 2


class RawExample(NamedTuple):
    text:           str
    bias_labels:    list[int]     # multi-hot, len=8
    lean_label:     int           # 0/1/2
    bias_intensity: float         # 0.0–1.0


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC CORPUS
# ─────────────────────────────────────────────────────────────────────────────

_LEFT_TEMPLATES = [
    ("Corporate greed is destroying the middle class as billionaires hoard unprecedented wealth.",
     [LL, FR], 0.78),
    ("Systemic racism continues to devastate marginalized communities across America.",
     [LL, FR, EP], 0.80),
    ("Experts warn that climate catastrophe threatens millions of innocent lives.",
     [LL, EP], 0.72),
    ("The progressive movement is fighting back against the alarming rise of inequality.",
     [LL, FR], 0.65),
    ("Some say the rich are destroying democracy with their shocking dark money influence.",
     [LL, EP, SE], 0.75),
    ("Brave activists are battling against corporate devastation of our environment.",
     [LL, FR], 0.70),
    ("The healthcare system is collapsing, leaving millions without coverage.",
     [LL, SE], 0.68),
    ("Heroic workers are standing up against unprecedented exploitation by the ruling class.",
     [LL, FR], 0.74),
    ("Everyone knows the wealthy elite are rigging the system against ordinary people.",
     [LL, EP], 0.77),
    ("Alarming new study shows climate change may catastrophically impact vulnerable populations.",
     [LL, EP, SE], 0.73),
    ("The desperate struggle of working families against corporate power continues.",
     [LL, FR], 0.66),
    ("Critics argue the government is failing marginalized communities in a stunning betrayal.",
     [LL, EP], 0.71),
    ("Shocking revelations about income inequality expose the rigged economic system.",
     [LL, SE], 0.76),
    ("Many believe the fossil fuel industry is destroying the planet for profit.",
     [LL, EP], 0.69),
    ("The unprecedented attack on workers' rights is devastating families nationwide.",
     [LL, FR, SE], 0.80),
]

_RIGHT_TEMPLATES = [
    ("The radical left's socialist agenda threatens to destroy American freedom and prosperity.",
     [LL, FR], 0.82),
    ("Government overreach is devastating small businesses and killing the American dream.",
     [LL, FR], 0.78),
    ("Experts warn the open border crisis is putting innocent Americans at risk.",
     [LL, EP, FR], 0.80),
    ("The mainstream media's shocking bias against conservatives is alarming.",
     [LL, SE], 0.75),
    ("Some say the deep state is waging a catastrophic war against traditional values.",
     [LL, EP, FR], 0.83),
    ("Brave patriots are fighting back against the unprecedented government tyranny.",
     [LL, FR], 0.77),
    ("The collapsing economy under failed liberal policies is devastating working families.",
     [LL, FR, SE], 0.79),
    ("Alarming crime surge in Democratic cities is putting millions of innocent lives at risk.",
     [LL, EP, FR, SE], 0.84),
    ("Everyone knows the globalist agenda is destroying national sovereignty.",
     [LL, EP], 0.76),
    ("The disastrous woke ideology is tearing apart the fabric of American society.",
     [LL, FR, SE], 0.81),
    ("Critics argue the administration's catastrophic energy policies are crippling America.",
     [LL, EP], 0.73),
    ("Many believe cancel culture is a dangerous attack on free speech.",
     [LL, EP, FR], 0.71),
    ("The stunning betrayal of border security is an outrageous failure of leadership.",
     [LL, SE], 0.78),
    ("Insiders reveal the shocking truth about the government's massive surveillance overreach.",
     [LL, EP, SE], 0.80),
    ("The unprecedented threat from illegal immigration demands immediate action.",
     [LL, FR, SE], 0.79),
]

_CENTER_TEMPLATES = [
    # Domestic/economic factual reporting
    ("The Federal Reserve held interest rates steady at 5.25% following its September meeting.",
     [], 0.05),
    ("Unemployment fell to 3.9% in Q3, the Bureau of Labor Statistics reported.",
     [], 0.04),
    ("The Senate passed the infrastructure bill 69–30 with bipartisan support.",
     [], 0.06),
    ("Scientists published a study in Nature linking carbon emissions to temperature rise.",
     [], 0.08),
    ("Company reported quarterly revenue of $4.2 billion, up 12% year-over-year.",
     [AN], 0.10),
    ("The Supreme Court ruled 6–3 in favor of the plaintiffs in the redistricting case.",
     [], 0.07),
    ("Health officials confirmed 143 cases of the new strain across seven states.",
     [], 0.05),
    ("The prime minister announced elections will be held on November 14th.",
     [], 0.04),
    ("Researchers found that the vaccine showed 94% efficacy in phase-3 trials.",
     [], 0.06),
    ("The city council voted 7–2 to approve the new zoning regulations.",
     [], 0.05),
    ("Stock prices ranged between $45 and $52 during Tuesday's trading session.",
     [AN], 0.09),
    ("According to census data, the population grew by 1.2% over the previous decade.",
     [EP], 0.12),
    ("The report found that average home prices increased by 8.3% in the past year.",
     [AN], 0.11),
    ("Officials say the new policy will take effect on January 1st of next year.",
     [], 0.06),
    ("The trade deficit narrowed to $68.9 billion, from $74.1 billion the prior month.",
     [AN], 0.08),
    ("Temperatures averaged 24 degrees Celsius during the summer months.",
     [], 0.03),
    ("The legislation includes provisions for both tax cuts and increased social spending.",
     [], 0.07),
    ("Scientists warn climate change may contribute to more frequent extreme weather events.",
     [EP], 0.15),
    # International / diplomatic journalism (neutral wire-service style)
    ("Peace talks between the two delegations resumed in Geneva following a ceasefire agreement.",
     [], 0.05),
    ("The foreign minister said in a statement that diplomatic efforts would continue.",
     [], 0.04),
    ("UN officials confirmed that humanitarian aid had reached the affected region.",
     [], 0.06),
    ("Both delegations met with mediators before the bilateral talks began on Monday.",
     [], 0.05),
    ("Ceasefire negotiations progressed after the security council convened an emergency session.",
     [], 0.07),
    ("Sources close to the negotiations said progress had been made on key conditions.",
     [EP], 0.12),
    ("The government spokesperson confirmed the announcement would be made Thursday.",
     [], 0.04),
    ("According to officials, the peace process is expected to continue through next month.",
     [EP], 0.08),
    ("The prime minister said further talks would depend on all parties fulfilling commitments.",
     [], 0.06),
    ("Mediators from the international community facilitated overnight sessions between both sides.",
     [], 0.05),
    ("The foreign ministry said the country would continue multilateral efforts for a resolution.",
     [], 0.05),
    ("Reporters confirmed the summit would include representatives from twelve countries.",
     [], 0.04),
    ("Officials confirmed the agreement was signed by both parties after weeks of negotiation.",
     [], 0.05),
    ("State department officials said the ceasefire was holding despite isolated incidents.",
     [], 0.06),
    ("Sources told reporters that a breakthrough in the peace talks was expected by the weekend.",
     [EP], 0.10),
]

_LOADED_LANG_TEMPLATES = [
    ("The heroic firefighters bravely battled the catastrophic blaze threatening thousands.",
     [LL, FR], CENTER, 0.72),
    ("This stunning breakthrough promises to revolutionize medicine forever.",
     [LL, SE], CENTER, 0.60),
    ("Devastating scandal rocks the industry as shocking new revelations emerge.",
     [LL, SE], CENTER, 0.73),
    ("Alarming new research suggests disturbing trends in youth mental health.",
     [LL, SE], CENTER, 0.65),
]

_ANCHORING_TEMPLATES = [
    ("The product costs just $99, slashed from the regular price of $499 — up to 80% savings.",
     [AN], CENTER, 0.35),
    ("As much as $2 trillion could be wiped from the economy if the deal fails.",
     [AN, SE], CENTER, 0.50),
    ("The minimum of $15 per hour wage would rise to $25, a maximum of 67% increase.",
     [AN], CENTER, 0.20),
    ("Starts at only $9.99 per month, down from the standard $29.99.",
     [AN], CENTER, 0.30),
]

_EPISTEMIC_TEMPLATES = [
    ("Some say the controversial decision may represent unprecedented government overreach.",
     [LL, EP], CENTER, 0.65),
    ("Reportedly, many believe the official statistics could be misleading.",
     [EP], CENTER, 0.45),
    ("Officials claim the policy might be effective, but experts suggest otherwise.",
     [EP], CENTER, 0.40),
    ("Insiders reveal what appears to be a systematic cover-up of the investigation.",
     [EP, SE], CENTER, 0.62),
    ("Everyone knows this is clearly the most important election of our lifetime.",
     [LL, EP], CENTER, 0.70),
    ("Sources indicate the administration may have allegedly violated regulations.",
     [EP], CENTER, 0.50),
]


_FALSE_BALANCE_TEMPLATES = [
    ("Both sides make valid points on climate change, and scientists are divided on the issue.", [FB], CENTER, 0.55),
    ("Some scientists say vaccines are safe while others raise concerns — the debate continues.", [FB, EP], CENTER, 0.60),
    ("The jury is still out on whether human activity causes warming, with equally valid perspectives.", [FB], CENTER, 0.58),
    ("We may never know the truth about election fraud as both sides present compelling evidence.", [FB, EP], CENTER, 0.62),
    ("Experts are divided on mask effectiveness and the science remains unclear to many.", [FB, EP], CENTER, 0.57),
    ("The so-called consensus on this topic is contested — no clear answer exists either way.", [FB, EP], RIGHT, 0.65),
    ("No reasonable person can say either party is wrong — both sides have legitimate points.", [FB], CENTER, 0.40),
    ("Reasonable people disagree on immigration; we may never know who is right on this issue.", [FB], CENTER, 0.45),
]

_WHATABOUTISM_TEMPLATES = [
    ("But what about when the previous administration did the same thing and nobody cared?", [WA], CENTER, 0.65),
    ("The real story the mainstream media won't tell you is what they're hiding about this.", [WA, SE], CENTER, 0.72),
    ("Where was the outrage when the other party passed this exact same policy three years ago?", [WA], CENTER, 0.60),
    ("What the media won't report: the untold story of how the opposition started this crisis.", [WA, EP], CENTER, 0.68),
    ("Nobody mentions the protesters were silent when their own side committed the same acts.", [WA, IG], CENTER, 0.63),
    ("The story being buried by the press is that the real culprit is the previous government.", [WA, EP], CENTER, 0.70),
    ("Why don't they talk about all the failures under the last administration?", [WA], RIGHT, 0.58),
    ("Follow the money — look at who stands to gain from suppressing this story.", [WA, EP], CENTER, 0.67),
]

_IN_GROUP_TEMPLATES = [
    ("Real Americans know the elites in Washington are destroying our way of life.", [IG, LL], RIGHT, 0.78),
    ("The silent majority is waking up to what the establishment has been doing for decades.", [IG, FR], RIGHT, 0.72),
    ("Working people versus the globalists: the battle for America's future has arrived.", [IG, FR], RIGHT, 0.80),
    ("The deep state is coming for ordinary Americans who just want to live their lives.", [IG, LL, SE], RIGHT, 0.82),
    ("True patriots stand with us against the radical agenda that threatens our values.", [IG, LL], RIGHT, 0.75),
    ("The forgotten people are being betrayed by the ruling class every single day.", [IG, FR], CENTER, 0.68),
    ("They want to destroy everything we hold dear — our families, our faith, our freedom.", [IG, LL, SE], RIGHT, 0.85),
    ("Corporate elites and their media allies are waging war on ordinary working families.", [IG, FR], LEFT, 0.73),
    ("The establishment doesn't care about us — they only protect their own power.", [IG, FR], CENTER, 0.65),
    ("The liberal mob attacks anyone who dares stand up for real American values.", [IG, LL, FR], RIGHT, 0.79),
]


def _build_example(text: str, active_indices: list[int], lean: int, intensity: float) -> RawExample:
    labels = [0] * NUM_BIAS
    for i in active_indices:
        if i < NUM_BIAS:
            labels[i] = 1
    return RawExample(text=text, bias_labels=labels, lean_label=lean, bias_intensity=intensity)


def build_synthetic_corpus(seed: int = 42) -> list[RawExample]:
    rng = random.Random(seed)
    corpus: list[RawExample] = []

    for text, indices, intensity in _LEFT_TEMPLATES:
        corpus.append(_build_example(text, indices, LEFT, intensity))

    for text, indices, intensity in _RIGHT_TEMPLATES:
        corpus.append(_build_example(text, indices, RIGHT, intensity))

    for text, indices, intensity in _CENTER_TEMPLATES:
        corpus.append(_build_example(text, indices, CENTER, intensity))

    for text, indices, lean, intensity in (
        _LOADED_LANG_TEMPLATES + _ANCHORING_TEMPLATES + _EPISTEMIC_TEMPLATES
        + _FALSE_BALANCE_TEMPLATES + _WHATABOUTISM_TEMPLATES + _IN_GROUP_TEMPLATES
    ):
        corpus.append(_build_example(text, indices, lean, intensity))

    # Augment: simple sentence reversal / case variation for robustness
    augmented = []
    for ex in corpus[:20]:
        words = ex.text.split()
        if len(words) > 6:
            rng.shuffle(words[2:5])
            aug_text = " ".join(words)
            augmented.append(RawExample(aug_text, ex.bias_labels, ex.lean_label, ex.bias_intensity))
    corpus.extend(augmented)

    rng.shuffle(corpus)
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# MBIC LOADER (optional real dataset)
# ─────────────────────────────────────────────────────────────────────────────

def load_mbic(path: str | Path | None = None) -> list[RawExample]:
    """
    Load the Media Bias In Context (MBIC) dataset.

    Place mbic.csv in data/raw/ — sourced from:
      Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE (Media-Bias-Group)

    Actual columns (semicolon-separated):
      text, news_link, outlet, topic, type (lean), group_id, num_sent,
      label_bias, label_opinion, article, biased_words
    Returns [] if file not found (training continues on synthetic data only).
    """
    csv_path = Path(path or DATA_DIR / "raw" / "mbic.csv")
    if not csv_path.exists():
        return []

    examples: list[RawExample] = []
    lean_map = {"left": LEFT, "center": CENTER, "right": RIGHT}

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            # Support both old column names and actual BABE column names
            text  = (row.get("text") or row.get("sentence") or "").strip()
            label = (row.get("label_bias") or row.get("label") or "").strip().lower()
            # 'type' column holds left/center/right lean in BABE format
            lean_raw = (row.get("type") or row.get("political_lean") or "").strip().lower()
            lean  = lean_map.get(lean_raw, CENTER)

            if not text or len(text) < 20:
                continue

            is_biased = int(label in ("biased", "1", "true"))
            # Map to our category format (MBIC has 'type' column)
            bias_type  = row.get("type", "").strip().lower()
            labels     = [0] * NUM_BIAS
            if is_biased:
                if "language" in bias_type or "word" in bias_type:
                    labels[LL] = 1
                if "frame" in bias_type:
                    labels[FR] = 1
                if "epistem" in bias_type or "hedg" in bias_type:
                    labels[EP] = 1
                if not any(labels):
                    labels[LL] = 1   # fallback: treat unknown biased as loaded language

            intensity = 0.7 if is_biased else 0.1
            examples.append(RawExample(text, labels, lean, float(intensity)))

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# QBIAS LOADER (AllSides-sourced articles)
# ─────────────────────────────────────────────────────────────────────────────

def load_qbias(path: str | Path | None = None, max_per_class: int = 5000) -> list[RawExample]:
    """
    Load Qbias dataset (allsides_balanced_news_headlines-texts.csv).
    Columns: title, tags, heading, source, text, bias_rating (left/center/right)
    max_per_class: cap per lean class to prevent imbalance.
    """
    csv_path = Path(path or DATA_DIR / "raw" / "qbias_articles.csv")
    if not csv_path.exists():
        return []

    examples: list[RawExample] = []
    lean_map = {"left": LEFT, "center": CENTER, "right": RIGHT}
    counts   = {LEFT: 0, CENTER: 0, RIGHT: 0}

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text        = (row.get("text") or "").strip()
            bias_rating = (row.get("bias_rating") or "center").strip().lower()
            lean        = lean_map.get(bias_rating, CENTER)

            if not text or len(text) < 50:
                continue
            if counts[lean] >= max_per_class:
                continue

            labels = [0] * NUM_BIAS
            if lean == LEFT:
                labels[LL] = 1
                labels[FR] = 1
                intensity  = 0.62
            elif lean == RIGHT:
                labels[LL] = 1
                labels[IG] = 1
                intensity  = 0.62
            else:
                intensity = 0.12

            counts[lean] += 1
            examples.append(RawExample(text[:1500], labels, lean, intensity))

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# SEMEVAL 2019 TASK 4 LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _strip_xml_tags(text: str) -> str:
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_semeval(
    articles_path:    str | Path | None = None,
    groundtruth_path: str | Path | None = None,
) -> list[RawExample]:
    """
    Load SemEval-2019 Task 4 byarticle dataset (XML).
    hyperpartisan=True  -> LL+SE labels, intensity=0.75
    hyperpartisan=False -> no labels, intensity=0.10
    """
    import xml.etree.ElementTree as ET
    base    = DATA_DIR / "raw" / "semeval"
    art_p   = Path(articles_path    or base / "articles-training-byarticle-20181122.xml")
    gt_p    = Path(groundtruth_path or base / "ground-truth-training-byarticle-20181122.xml")

    if not art_p.exists() or not gt_p.exists():
        return []

    gt_map: dict[str, bool] = {}
    try:
        for art in ET.parse(gt_p).getroot():
            gt_map[art.get("id", "")] = art.get("hyperpartisan", "false").lower() == "true"
    except ET.ParseError:
        return []

    examples: list[RawExample] = []
    try:
        for art in ET.parse(art_p).getroot():
            art_id = art.get("id", "")
            if art_id not in gt_map:
                continue
            title   = art.get("title", "")
            body    = ET.tostring(art, encoding="unicode", method="text")
            text    = _strip_xml_tags(f"{title}. {body}")[:1500]
            if len(text) < 50:
                continue
            is_hyper = gt_map[art_id]
            labels   = [0] * NUM_BIAS
            if is_hyper:
                labels[LL] = 1
                labels[SE] = 1
                intensity  = 0.75
            else:
                intensity = 0.10
            examples.append(RawExample(text, labels, CENTER, intensity))
    except ET.ParseError:
        return []

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BiasDataset(Dataset):

    def __init__(
        self,
        examples:  list[RawExample],
        tokenizer: PreTrainedTokenizerFast,
        max_len:   int = MODEL_MAX_LENGTH,
    ) -> None:
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex  = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "bias_labels":    torch.tensor(ex.bias_labels, dtype=torch.float),
            "lean_label":     torch.tensor(ex.lean_label,  dtype=torch.long),
            "bias_intensity": torch.tensor(ex.bias_intensity, dtype=torch.float),
        }


def _stratified_sample(
    examples: list[RawExample],
    max_total: int,
    seed: int,
) -> list[RawExample]:
    """Cap dataset to max_total while preserving lean-class balance."""
    if max_total <= 0 or len(examples) <= max_total:
        return examples
    buckets: dict[int, list[RawExample]] = {LEFT: [], CENTER: [], RIGHT: []}
    for ex in examples:
        buckets[ex.lean_label].append(ex)
    per_class = max_total // 3
    rng = random.Random(seed)
    sampled: list[RawExample] = []
    for cls_examples in buckets.values():
        rng.shuffle(cls_examples)
        sampled.extend(cls_examples[:per_class])
    rng.shuffle(sampled)
    return sampled


def build_loaders(
    tokenizer: PreTrainedTokenizerFast,
    val_split:   float = 0.15,
    batch_size:  int   = 4,
    seed:        int   = 42,
    max_length:  int   = 256,
    max_samples: int   = 2000,
) -> tuple:
    """
    Build train + validation DataLoaders from all available sources.
    Returns (train_loader, val_loader, class_weights_tensor).
    max_samples: stratified cap on total examples (0 = no cap).
    """
    synthetic = build_synthetic_corpus(seed)
    mbic      = load_mbic()
    qbias     = load_qbias()
    # SemEval excluded: it's a hyperpartisan dataset with no real lean labels
    # (all examples were force-labeled CENTER, which poisons the lean head)

    print(f"Dataset: synthetic={len(synthetic)}, mbic={len(mbic)}, "
          f"qbias={len(qbias)} "
          f"-> total={len(synthetic)+len(mbic)+len(qbias)}")

    all_examples = synthetic + mbic + qbias
    rng = random.Random(seed)
    rng.shuffle(all_examples)

    if max_samples > 0:
        all_examples = _stratified_sample(all_examples, max_samples, seed)
        print(f"Stratified subsample -> {len(all_examples)} examples "
              f"(~{len(all_examples)//3} per lean class)")

    n_val   = max(1, int(len(all_examples) * val_split))
    n_train = len(all_examples) - n_val
    train_ex, val_ex = all_examples[:n_train], all_examples[n_train:]

    train_ds = BiasDataset(train_ex, tokenizer, max_len=max_length)
    val_ds   = BiasDataset(val_ex,   tokenizer, max_len=max_length)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights for BCE: inverse-frequency, clamped to avoid explosion
    pos_counts    = torch.zeros(NUM_BIAS)
    for ex in train_ex:
        pos_counts += torch.tensor(ex.bias_labels, dtype=torch.float)
    neg_counts    = len(train_ex) - pos_counts
    class_weights = (neg_counts / (pos_counts + 1e-6)).clamp(max=10.0)

    return train_loader, val_loader, class_weights
