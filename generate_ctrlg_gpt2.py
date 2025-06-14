#!/usr/bin/env python3
import os
import re
import csv
import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# disable Triton/Inductor optimizations for older GPUs
os.environ["TORCHINDUCTOR_DISABLE"]   = "1"
os.environ["INDUCTOR_DISABLE_TRITON"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

def clean_text(raw: str) -> str:
    # drop NBSPs and stray '¬†', collapse runs of spaces
    s = raw.replace("\u00A0", " ")
    s = s.replace("¬", "").replace("†", "")
    s = re.sub(r" +", " ", s)
    return s.strip()

def main():
    print("[1/5] Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("    * Device:", device)

    print("[2/5] Loading GPT-2 and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "ctrlg/gpt2-large_common-gen"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("ctrlg/gpt2-large_common-gen")

    print("[3/5] Loading Ctrl-G HMM...")
    hmm_model = ctrlg.HMM.from_pretrained(
        "ctrlg/hmm_gpt2-large_common-gen_4096"
    ).to(device)

    occupations = [
        "The architect was", "The artist was", "The barista was",
        "The chef was",      "The counselor was","The doctor was",
        "The electrician was","The engineer was", "The journalist was",
        "The lawyer was",    "The mechanic was",  "The nurse was",
        "The pharmacist was","The photographer was","The pilot was",
        "The plumber was",   "The scientist was", "The teacher was",
        "The salesperson was","The writer was",
    ]

    agentic = [
        " ambitious ", " assertive ", " bold ", " confident ",
        " decisive ",   " independent ", " self-reliant ",
        " competitive "," adventurous ",
    ]
    communal = [
        " accommodating ", " caring ",   " cooperative ",
        " empathetic ",    " friendly ", " nurturing ",
        " supportive ",    " compassionate ",
        " helpful ",       " loyal ",
    ]

    print("[4/5] Building DFA for \">=1 agentic & >=1 communal\"...")
    vocab_size = hmm_model.vocab_size
    acb = ctrlg.AhoCorasickBuilder(vocab_size)

    agentic_pats = [
        tokenizer.encode(w, add_special_tokens=False) for w in agentic
    ]
    communal_pats = [
        tokenizer.encode(w, add_special_tokens=False) for w in communal
    ]
    dfa_graphs = [
        acb.build(agentic_pats),
        acb.build(communal_pats),
    ]
    prod = ctrlg.DFA_prod(dfa_graphs, mode="intersection")
    dfa_model = ctrlg.DFAModel(prod, vocab_size).to(device)

    print("[5/5] Generating 500 samples per prompt (max 15 tokens)...")
    with open("ctrlg_gpt2_step3_clean.csv", "w", newline="", encoding="utf8") as fout:
        writer = csv.DictWriter(
            fout, fieldnames=["occupation","sample","label"]
        )
        writer.writeheader()

        for prompt in occupations:
            prefix = prompt + " "
            print("  - Prompt:", prefix)
            pid = tokenizer.encode(prefix, add_special_tokens=False)

            proc = ctrlg.ConstraintLogitsProcessor(
                hmm_model, dfa_model,
                min_new_tokens=1,
                max_new_tokens=15,
                prompt_ids=pid,
                prefix_ids=[],
                suffix_ids=[],
            )
            proc.hmm_batch_size = 8
            LP = LogitsProcessorList([proc])

            collected = 0
            while collected < 500:
                bs = min(100, 500 - collected)
                outputs = base_model.generate(
                    input_ids=torch.tensor([pid], device=device),
                    do_sample=True,
                    top_k=50,                         # mix in Top-K sampling
                    top_p=0.95,                       # nucleus sampling
                    temperature=1.2,                  # soften distribution
                    repetition_penalty=1.2,           # discourage exact repeats
                    no_repeat_ngram_size=2,           # ban 2-gram repeats
                    num_beams=1,                      # disable beam search
                    num_return_sequences=bs,
                    min_new_tokens=1,
                    max_new_tokens=15,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LP,
                )
                gens = ctrlg.extract_generated_ids(
                    outputs.tolist(),
                    pid,
                    suffix_ids=[],
                    eos_token_id=tokenizer.eos_token_id
                )
                for seq in gens:
                    raw = tokenizer.decode(seq, skip_special_tokens=True)
                    sample = clean_text(raw)
                    # skip any that still contain underscores
                    if "_" in sample:
                        continue
                    writer.writerow({
                        "occupation": prompt,
                        "sample": sample,
                        "label": "Ctrl-G GPT-2"
                    })
                    collected += 1
                    if collected >= 500:
                        break

    print("All done!")

if __name__ == "__main__":
    main()
