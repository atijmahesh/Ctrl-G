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
    return re.sub(r" +", " ", s).strip()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GPT2-large and tokenizer (domain-adapted to CommonGen)
    base_model = AutoModelForCausalLM.from_pretrained(
        "ctrlg/gpt2-large_common-gen"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("ctrlg/gpt2-large_common-gen")

    # Load the ideal 32 768-state HMM (per Commonsense Gen experiments)
    hmm_model = ctrlg.HMM.from_pretrained(
        "ctrlg/hmm_gpt2-large_common-gen_32768"
    ).to(device)

    # Prepare prompts
    occupations = [
        "The architect was", "The artist was", "The barista was",
        "The chef was",      "The counselor was","The doctor was",
        "The electrician was","The engineer was", "The journalist was",
        "The lawyer was",    "The mechanic was",  "The nurse was",
        "The pharmacist was","The photographer was","The pilot was",
        "The plumber was",   "The scientist was", "The teacher was",
        "The salesperson was","The writer was",
    ]

    # Build DFAs for agentic+communal constraints
    agentic = [" ambitious ", " assertive ", " bold ", " confident ",
               " decisive ",   " independent ", " self-reliant ",
               " competitive "," adventurous "]
    communal = [" accommodating ", " caring ",   " cooperative ",
                " empathetic ",    " friendly ", " nurturing ",
                " supportive ",    " compassionate ",
                " helpful ",       " loyal "]

    acb = ctrlg.AhoCorasickBuilder(hmm_model.vocab_size)
    agentic_pats = [tokenizer.encode(w, add_special_tokens=False) for w in agentic]
    communal_pats = [tokenizer.encode(w, add_special_tokens=False) for w in communal]
    dfa_graphs = [acb.build(agentic_pats), acb.build(communal_pats)]
    prod = ctrlg.DFA_prod(dfa_graphs, mode="intersection")
    dfa_model = ctrlg.DFAModel(prod, hmm_model.vocab_size).to(device)

    # Sampling settings
    MIN_TOK, MAX_TOK = 1, 15

    print(f"Generating 500 samples per prompt (tokens ∈ [{MIN_TOK}, {MAX_TOK}])...")
    with open("ctrlg_gpt2_step3_clean.csv", "w", newline="", encoding="utf8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["occupation","sample","label"])
        writer.writeheader()

        for prompt in occupations:
            prefix = prompt + " "
            pid = tokenizer.encode(prefix, add_special_tokens=False)

            # Initialize constraint processor once per prompt
            proc = ctrlg.ConstraintLogitsProcessor(
                hmm_model, dfa_model,
                MIN_TOK, MAX_TOK,
                prompt_ids=pid,
                prefix_ids=[],
                suffix_ids=[],
            )

            collected = 0
            while collected < 500:
                # Batch size up to 100 to match paper’s evaluation protocol
                bs = min(100, 500 - collected)
                proc.hmm_batch_size = bs  # match HMM inference to generation batch

                LP = LogitsProcessorList([proc])
                outputs = base_model.generate(
                    input_ids=torch.tensor([pid], device=device),
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.2,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    num_beams=1,
                    num_return_sequences=bs,
                    min_new_tokens=MIN_TOK,
                    max_new_tokens=MAX_TOK,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LP,
                )

                # Extract & clean
                gens = ctrlg.extract_generated_ids(
                    outputs.tolist(),
                    pid,
                    suffix_ids=[],
                    eos_token_id=tokenizer.eos_token_id
                )
                for seq in gens:
                    raw = tokenizer.decode(seq, skip_special_tokens=True)
                    sample = clean_text(raw)
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