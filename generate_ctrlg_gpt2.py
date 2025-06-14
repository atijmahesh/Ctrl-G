#!/usr/bin/env python3
import os
import re
import csv
import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# Turn off advanced optimizations
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["INDUCTOR_DISABLE_TRITON"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_text(raw: str) -> str:
    # Remove nonbreaking spaces and stray chars, collapse spaces
    s = raw.replace("\u00A0", " ")
    s = s.replace("¬", "").replace("†", "")
    return re.sub(r" +", " ", s).strip()

def main():
    print("STARTING SCRIPT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading GPT2 model and tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        "ctrlg/gpt2-large_common-gen_4096"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("ctrlg/gpt2-large_common-gen")

    print("Loading HMM with 4096 states")
    hmm = ctrlg.HMM.from_pretrained(
        "ctrlg/hmm_gpt2-large_common-gen_32768"
    ).to(device)
    print("HMM states:", hmm.num_states)

    # Build DFA
    patterns_agentic = [" ambitious ", " assertive ", " bold ", " confident ",
                        " decisive ", " independent ", " self-reliant ",
                        " competitive ", " adventurous "]
    patterns_communal = [" accommodating ", " caring ", " cooperative ",
                         " empathetic ", " friendly ", " nurturing ",
                         " supportive ", " compassionate ",
                         " helpful ", " loyal "]

    print("Building DFA for constraints")
    acb = ctrlg.AhoCorasickBuilder(hmm.vocab_size)
    pats_a = [tokenizer.encode(w, add_special_tokens=False) for w in patterns_agentic]
    pats_c = [tokenizer.encode(w, add_special_tokens=False) for w in patterns_communal]
    dfa_a = acb.build(pats_a)
    dfa_c = acb.build(pats_c)
    prod = ctrlg.DFA_prod([dfa_a, dfa_c], mode="intersection")
    dfa = ctrlg.DFAModel(prod, hmm.vocab_size).to(device)

    MIN_TOK = 1
    MAX_TOK = 15
    occupations = [
        "The architect was", "The artist was", "The barista was",
        "The chef was", "The counselor was", "The doctor was",
        "The electrician was", "The engineer was", "The journalist was",
        "The lawyer was", "The mechanic was", "The nurse was",
        "The pharmacist was", "The photographer was", "The pilot was",
        "The plumber was", "The scientist was", "The teacher was",
        "The salesperson was", "The writer was"
    ]

    with open("output.csv", "w", newline="", encoding="utf8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["occupation", "sample", "label"])
        writer.writeheader()

        for idx, prompt in enumerate(occupations, start=1):
            print(f"Prompt {idx}/{len(occupations)}: {prompt}")
            prefix_ids = tokenizer.encode(prompt + " ", add_special_tokens=False)
            proc = ctrlg.ConstraintLogitsProcessor(
                hmm, dfa, MIN_TOK, MAX_TOK,
                prompt_ids=prefix_ids,
                prefix_ids=[],
                suffix_ids=[]
            )

            collected = 0
            batch = 0
            while collected < 500:
                batch += 1
                bs = min(100, 500 - collected)
                proc.hmm_batch_size = bs
                print(f"  Batch {batch}: sampling {bs} (collected {collected})")

                outputs = model.generate(
                    input_ids=torch.tensor([prefix_ids], device=device),
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.0,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    num_return_sequences=bs,
                    min_new_tokens=MIN_TOK,
                    max_new_tokens=MAX_TOK,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([proc])
                )

                gens = ctrlg.extract_generated_ids(
                    outputs.tolist(),
                    prefix_ids,
                    suffix_ids=[],
                    eos_token_id=tokenizer.eos_token_id
                )
                for seq in gens:
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    sample = clean_text(text)
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

            print(f"  Done {prompt}: collected {collected} samples\n")

    print("ALL DONE")

if __name__ == "__main__":
    main()