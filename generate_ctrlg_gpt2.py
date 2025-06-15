#!/usr/bin/env python3
import os
import re
import csv
import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# Configuration
OCCUPATIONS = [
    "The architect was", "The artist was", "The barista was", "The chef was",
    "The counselor was", "The doctor was", "The electrician was",
    "The engineer was", "The journalist was", "The lawyer was",
    "The mechanic was", "The nurse was", "The pharmacist was",
    "The photographer was", "The pilot was", "The plumber was",
    "The scientist was", "The teacher was", "The salesperson was",
    "The writer was"
]

AGENTIC = [
    " ambitious", " assertive", " bold", " confident", " decisive",
    " independent", " self-reliant", " competitive", " adventurous", " dominant"
]
COMMUNAL = [
    " accommodating", " caring", " cooperative", " empathetic", " friendly",
    " nurturing", " supportive", " compassionate", " helpful", " loyal"
]

NUM_SAMPLES        = 500
TEMP, TOP_P        = 1.0, 0.95
MIN_TOK, MAX_TOK   = 2, 20

OUTPUT_CSV = "ctrlg_gpt2_step3.csv"

def clean_text(text):
    text = text.replace("\u00A0", " ").replace("¬", "").replace("†", "")
    return re.sub(r" +", " ", text).strip()

def step3_ctrlg_gpt2():
    print("STARTING Ctrl-G GPT-2 Step 3 generation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 3.1 Load model and tokenizer
    print("Loading GPT-2 model and tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        "ctrlg/gpt2-large_common-gen"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("ctrlg/gpt2-large_common-gen")
    print("Model and tokenizer loaded")

    # 3.1 Load HMM
    print("Loading Ctrl-G HMM")
    hmm = ctrlg.HMM.from_pretrained(
        "ctrlg/hmm_gpt2-large_common-gen_4096"
    ).to(device)
    print("HMM loaded")

    # 3.2 Build DFA requiring at least one agentic term and one communal term
    print("Building DFA with agentic and communal constraints")
    vs = hmm.vocab_size
    acb = ctrlg.AhoCorasickBuilder(vs)
    pats_a = [tokenizer.encode(w, add_special_tokens=False) for w in AGENTIC]
    pats_c = [tokenizer.encode(w, add_special_tokens=False) for w in COMMUNAL]

    prod = ctrlg.DFA_prod([acb.build(pats_a), acb.build(pats_c)], mode="intersection")
    dfa  = ctrlg.DFAModel(prod, vs).to(device)
    print("DFA built successfully")

    # 3.3 Generate samples
    period_id = tokenizer.encode(".", add_special_tokens=False)[0]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf8", buffering=1) as fout:
        writer = csv.DictWriter(fout, fieldnames=["occupation", "sample", "label"])
        writer.writeheader()

        for occ in OCCUPATIONS:
            print(f"Generating for occupation: {occ}")
            prefix_ids = tokenizer.encode(occ, add_special_tokens=False)

            collected = 0
            batch_num = 0
            batch_size = 100

            while collected < NUM_SAMPLES:
                batch_num += 1
                bs = min(batch_size, NUM_SAMPLES - collected)
                print(f"Batch {batch_num}: sampling {bs}, collected so far {collected}")

                proc = ctrlg.ConstraintLogitsProcessor(
                    hmm, dfa, MIN_TOK, MAX_TOK,
                    prompt_ids=prefix_ids,
                    prefix_ids=[],           # only DFA enforces constraints
                    suffix_ids=[period_id]   # stop at first period
                )
                proc.hmm_batch_size = bs

                outputs = model.generate(
                    input_ids=torch.tensor([prefix_ids], device=device),
                    do_sample=True,
                    temperature=TEMP,
                    top_p=TOP_P,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    min_new_tokens=MIN_TOK,
                    max_new_tokens=MAX_TOK,
                    eos_token_id=period_id,
                    pad_token_id=tokenizer.eos_token_id,
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
                    writer.writerow({
                        "occupation": occ,
                        "sample":     sample,
                        "label":      "Ctrl-G GPT-2"
                    })
                    collected += 1
                    if collected % 50 == 0:
                        print(f"Sample number {collected} for '{occ}': {sample}")
                    if collected % 100 == 0:
                        print(f"Collected {collected} of {NUM_SAMPLES} for '{occ}'")
                    if collected >= NUM_SAMPLES:
                        break

            print(f"Completed {collected} samples for occupation: {occ}\n")

    print("All occupations done. Results saved to", OUTPUT_CSV)

if __name__ == "__main__":
    step3_ctrlg_gpt2()
