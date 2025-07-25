#!/usr/bin/env python3
import os
import re
import csv
import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
OCCUPATIONS = [
    "architect", "artist", "barista", "chef", "counselor",
    "doctor", "electrician", "engineer", "journalist", "lawyer",
    "mechanic", "nurse", "pharmacist", "photographer", "pilot",
    "plumber", "scientist", "teacher", "salesperson", "writer"
]

AGENTIC = [
    "ambitious", "assertive", "bold", "confident", "decisive",
    "independent", "self-reliant", "competitive", "adventurous", "dominant"
]
COMMUNAL = [
    "accommodating", "caring", "cooperative", "empathetic", "friendly",
    "nurturing", "supportive", "compassionate", "helpful", "loyal"
]

NUM_SAMPLES        = 500
TEMP, TOP_P        = 1.0, 0.95
MIN_WORDS, MAX_WORDS = 8, 15        # word-length constraints
OUTPUT_CSV         = "ctrlg_prefix_completions.csv"

# ─── UTILITIES ───────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.replace("\u00A0", " ").replace("¬", "").replace("†", "")
    return re.sub(r"\s+", " ", text).strip()

def count_words(text: str) -> int:
    return len(text.strip().split())

# ─── MAIN ─────────────────────────────────────────────────────────────────────────
def run_prefix_ctrlg():
    print("STARTING prefix-only Ctrl-G generation", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    # 1) Load model + tokenizer + HMM
    print("Loading model, tokenizer, and HMM...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "ctrlg/gpt2-large_common-gen"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("ctrlg/gpt2-large_common-gen")
    hmm = ctrlg.HMM.from_pretrained(
        "ctrlg/hmm_gpt2-large_common-gen_4096"
    ).to(device)
    print("Loaded.", flush=True)

    # 2) Build DFA for ≥1 agentic & ≥1 communal
    vs = hmm.vocab_size
    acb = ctrlg.AhoCorasickBuilder(vs)
    pats_a = [tokenizer.encode(" " + w, add_special_tokens=False) for w in AGENTIC]
    pats_c = [tokenizer.encode(" " + w, add_special_tokens=False) for w in COMMUNAL]
    prod = ctrlg.DFA_prod([acb.build(pats_a), acb.build(pats_c)], mode="intersection")
    dfa  = ctrlg.DFAModel(prod, vs).to(device)
    print("DFA built.", flush=True)

    # 3) Prepare stop token
    period_id = tokenizer.encode(".", add_special_tokens=False)[0]

    # 4) Open CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf8", buffering=1) as fout:
        writer = csv.DictWriter(fout, fieldnames=["occupation", "sample", "label"])
        writer.writeheader()

        total_runs = len(OCCUPATIONS) * NUM_SAMPLES
        run_counter = 0

        for occ in OCCUPATIONS:
            prefix = f"The {occ} was"
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            print(f"\n=== Occupation: {occ} ===", flush=True)

            collected = 0
            batch_size = 100

            while collected < NUM_SAMPLES:
                bs = min(batch_size, NUM_SAMPLES - collected)
                proc = ctrlg.ConstraintLogitsProcessor(
                    hmm, dfa,
                    min_new_tokens=MIN_WORDS,
                    max_new_tokens=MAX_WORDS,
                    prompt_ids=[],        # no instruction prompt
                    prefix_ids=prefix_ids,
                    suffix_ids=[period_id]
                )
                proc.hmm_batch_size = bs

                outputs = model.generate(
                    input_ids=torch.tensor([prefix_ids], device=device),
                    do_sample=True,
                    temperature=TEMP,
                    top_p=TOP_P,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    min_new_tokens=MIN_WORDS,
                    max_new_tokens=MAX_WORDS,
                    eos_token_id=period_id,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=bs,
                    logits_processor=LogitsProcessorList([proc])
                )

                gens = ctrlg.extract_generated_ids(
                    outputs.tolist(),
                    prefix_ids,
                    suffix_ids=[period_id],
                    eos_token_id=tokenizer.eos_token_id
                )

                for seq in gens:
                    text = clean_text(tokenizer.decode(seq, skip_special_tokens=True))
                    wc = count_words(text)
                    label = "Ctrl-G GPT-2"
                    if not (MIN_WORDS <= wc <= MAX_WORDS):
                        label += " [length OOB]"

                    writer.writerow({"occupation": occ, "sample": text, "label": label})
                    collected += 1
                    run_counter += 1

                    if run_counter % 50 == 0:
                        print(f"Run {run_counter}/{total_runs}: {text}", flush=True)

    print(f"\nAll occupations done. Results saved to '{OUTPUT_CSV}'", flush=True)

if __name__ == "__main__":
    run_prefix_ctrlg()
