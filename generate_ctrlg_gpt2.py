#!/usr/bin/env python3
import os
# ─── ENV BEFORE ANY TORCH IMPORT ──────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"]    = "0"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["TORCHINDUCTOR_DISABLE"]   = "1"
os.environ["INDUCTOR_DISABLE_TRITON"] = "1"

import re
import torch
import ctrlg
import csv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

class BanUnderscore(LogitsProcessor):
    """LogitsProcessor to ban the '_' token entirely."""
    def __init__(self, underscore_id: int):
        self.underscore_id = underscore_id

    def __call__(self, input_ids, scores):
        # set logit for '_' to -inf so it's never sampled
        scores[:, self.underscore_id] = -float("inf")
        return scores

def clean_text(raw: str) -> str:
    """Remove NBSPs, stray control chars, underscores, collapse whitespace."""
    # replace non-breaking space with normal space
    s = raw.replace("\u00A0", " ")
    # strip out any literal '¬' or '†' that crept in
    s = s.replace("¬", "").replace("†", "")
    # collapse any runs of whitespace (including tabs/newlines)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def main():
    # 1) Setup device
    print("[1/5] Setting up device and environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("* Device:", device)

    # 2) Load models
    BASE_MODEL_PATH = "ctrlg/gpt2-large_common-gen"
    HMM_MODEL_PATH  = "ctrlg/hmm_gpt2-large_common-gen_4096"

    print("[2/5] Loading GPT-2 checkpoint from", BASE_MODEL_PATH, "...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device).eval()
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    print("[3/5] Loading HMM from", HMM_MODEL_PATH, "...")
    hmm_model = ctrlg.HMM.from_pretrained(HMM_MODEL_PATH).to(device)

    # 3) Prompts + synonym lists
    occupations = [
        "The architect was", "The artist was", "The barista was", "The chef was",
        "The counselor was", "The doctor was", "The electrician was", "The engineer was",
        "The journalist was", "The lawyer was", "The mechanic was", "The nurse was",
        "The pharmacist was", "The photographer was", "The pilot was", "The plumber was",
        "The scientist was", "The teacher was", "The salesperson was", "The writer was"
    ]
    agentic = [
        " ambitious ", " assertive ", " bold ", " confident ", " decisive ",
        " independent ", " self-reliant ", " competitive ", " adventurous ", " assertive "
    ]
    communal = [
        " accommodating ", " caring ", " cooperative ", " empathetic ", " friendly ",
        " nurturing ", " supportive ", " compassionate ", " helpful ", " loyal "
    ]

    # 4) Build DFA constraint
    print("[4/5] Building DFA constraint (>=1 agentic + >=1 communal, 8-15 words)...")
    vocab_size = hmm_model.vocab_size
    ac_builder = ctrlg.AhoCorasickBuilder(vocab_size)
    wc_builder = ctrlg.WordCountBuilder(tokenizer, vocab_size)

    agentic_pats  = [tokenizer.encode(x, add_special_tokens=False) for x in agentic]
    communal_pats = [tokenizer.encode(x, add_special_tokens=False) for x in communal]

    dfa_graphs = [
        ac_builder.build(agentic_pats),
        ac_builder.build(communal_pats),
        wc_builder.build(8, 15)
    ]
    dfa_prod  = ctrlg.DFA_prod(dfa_graphs, mode="intersection")
    dfa_model = ctrlg.DFAModel(dfa_prod, vocab_size).to(device)

    # prepare BanUnderscore
    underscore_id = tokenizer.convert_tokens_to_ids("_")
    ban_us = BanUnderscore(underscore_id)

    # 5) Generate + write CSV
    OUTPUT_CSV = "ctrlg_gpt2_step3_clean.csv"
    print("[5/5] Generating samples and writing to", OUTPUT_CSV, "...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf8") as outf:
        writer = csv.DictWriter(outf, fieldnames=["occupation","sample","label"])
        writer.writeheader()

        for prompt in occupations:
            prompt_text = prompt + " "
            print("Prompt:", prompt_text)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            # set up constraint
            constraint = ctrlg.ConstraintLogitsProcessor(
                hmm_model, dfa_model,
                min_new_tokens=8, max_new_tokens=15,
                prompt_ids=prompt_ids,
                prefix_ids=[], suffix_ids=[]
            )
            constraint.hmm_batch_size = 8

            procs = LogitsProcessorList([constraint, ban_us])

            total, batch_size = 500, 100
            collected = 0
            while collected < total:
                bs = min(batch_size, total - collected)
                print(f"  sampling batch {collected+1}-{collected+bs} ...")
                out = base_model.generate(
                    input_ids=torch.tensor([prompt_ids], device=device),
                    do_sample=True,
                    top_p=0.95,
                    temperature=1.0,
                    num_beams=1,
                    num_return_sequences=bs,
                    min_new_tokens=8,
                    max_new_tokens=15,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=procs,
                )
                gen_ids = ctrlg.extract_generated_ids(
                    out.tolist(), prompt_ids, suffix_ids=[], eos_token_id=tokenizer.eos_token_id
                )

                for seq in gen_ids:
                    raw = tokenizer.decode(seq, skip_special_tokens=True)
                    clean = clean_text(raw)
                    writer.writerow({
                        "occupation": prompt,
                        "sample":    clean,
                        "label":     "Ctrl-G GPT-2"
                    })
                collected += bs

    print("All done!")

if __name__ == "__main__":
    main()