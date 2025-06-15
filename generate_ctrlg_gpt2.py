#!/usr/bin/env python3
import os
import re
import csv
import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
os.environ["TORCHINDUCTOR_DISABLE"]   = "1"
os.environ["INDUCTOR_DISABLE_TRITON"]  = "1"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"

# ─── Config ────────────────────────────────────────────────────────────
OUTPUT_CSV   = "ctrlg_gpt2_outputs.csv"
NUM_SAMPLES  = 500
TEMP, TOP_P  = 1.0, 0.95
MIN_TOK, MAX_TOK = 8, 15

OCCUPATIONS = [
    "The architect was", "The artist was", "The barista was", "The chef was",
    "The counselor was", "The doctor was", "The electrician was",
    "The engineer was", "The journalist was", "The lawyer was",
    "The mechanic was", "The nurse was", "The pharmacist was",
    "The photographer was", "The pilot was", "The plumber was",
    "The scientist was", "The teacher was", "The salesperson was",
    "The writer was"
]

# Pad each synonym with spaces for whole-word matching
AGENTIC = [
    " ambitious ", " assertive ", " bold ", " confident ", " decisive ",
    " independent ", " self-reliant ", " competitive ", " adventurous ",
    " dominant "
]
COMMUNAL = [
    " accommodating ", " caring ", " cooperative ", " empathetic ", " friendly ",
    " nurturing ", " supportive ", " compassionate ", " helpful ", " loyal "
]

# ─── Helpers ────────────────────────────────────────────────────────────
def clean_text(text):
    t = text.replace("\u00A0", " ").replace("¬", "").replace("†", "")
    return re.sub(r" +", " ", t).strip()

# ─── Ctrl-G GPT-2 Generation ─────────────────────────────────────────────
def run_ctrlg_gpt2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load GPT-2 + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "ctrlg/gpt2-large_common-gen"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("ctrlg/gpt2-large_common-gen")

    # 2) Load HMM
    hmm = ctrlg.HMM.from_pretrained(
        "ctrlg/hmm_gpt2-large_common-gen_4096"
    ).to(device)

    # 3) Build DFA requiring ≥1 agentic & ≥1 communal
    vs  = hmm.vocab_size
    acb = ctrlg.AhoCorasickBuilder(vs)
    pats_a = [tokenizer.encode(w, add_special_tokens=False) for w in AGENTIC]
    pats_c = [tokenizer.encode(w, add_special_tokens=False) for w in COMMUNAL]
    prod   = ctrlg.DFA_prod([acb.build(pats_a), acb.build(pats_c)], mode="intersection")
    dfa    = ctrlg.DFAModel(prod, vs).to(device)

    # 4) Period token to stop at first “.”
    period_id = tokenizer.encode(".", add_special_tokens=False)[0]

    # 5) Open CSV for writing
    with open(OUTPUT_CSV, "w", newline="", encoding="utf8", buffering=1) as fout:
        writer = csv.DictWriter(fout, fieldnames=["occupation","sample","label"])
        writer.writeheader()

        # 6) Loop through occupations
        for occ in OCCUPATIONS:
            print(f"> Generating for: {occ}")
            prefix_ids = tokenizer.encode(occ, add_special_tokens=False)

            collected = 0
            seen = set()
            batch_size = 100

            # 7) Sample until we have NUM_SAMPLES uniques
            while collected < NUM_SAMPLES:
                bs = min(batch_size, NUM_SAMPLES - collected)

                proc = ctrlg.ConstraintLogitsProcessor(
                    hmm, dfa, MIN_TOK, MAX_TOK,
                    prompt_ids=prefix_ids,
                    prefix_ids=[],            # only DFA enforces constraints
                    suffix_ids=[period_id]    # stop at period
                )
                proc.hmm_batch_size = bs

                outputs = model.generate(
                    input_ids=torch.tensor([prefix_ids], device=device),
                    do_sample=True,
                    top_p=TOP_P,
                    temperature=TEMP,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    min_new_tokens=MIN_TOK,
                    max_new_tokens=MAX_TOK,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=period_id,
                    num_return_sequences=bs,
                    num_beams=1,
                    logits_processor=LogitsProcessorList([proc])
                )

                gens = ctrlg.extract_generated_ids(
                    outputs.tolist(),
                    prefix_ids,
                    suffix_ids=[period_id],
                    eos_token_id=tokenizer.eos_token_id
                )
                for seq in gens:
                    sample = clean_text(tokenizer.decode(seq, skip_special_tokens=True))
                    if sample in seen:
                        continue
                    seen.add(sample)
                    writer.writerow({
                        "occupation": occ,
                        "sample":     sample,
                        "label":      "Ctrl-G GPT-2"
                    })
                    collected += 1
                    if collected % 100 == 0:
                        print(f"  → Collected {collected}/{NUM_SAMPLES}")
                    if collected >= NUM_SAMPLES:
                        break

    print("Done! Results in", OUTPUT_CSV)

if __name__ == "__main__":
    run_ctrlg_gpt2()
