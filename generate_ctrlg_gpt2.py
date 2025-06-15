#!/usr/bin/env python3
import os
import re
import csv
import argparse
import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# Turn off advanced optimizations
os.environ["TORCHINDUCTOR_DISABLE"]   = "1"
os.environ["INDUCTOR_DISABLE_TRITON"]  = "1"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"

def clean_text(raw: str) -> str:
    # Remove nonbreaking spaces and stray chars, collapse spaces
    s = raw.replace("\u00A0", " ")
    s = s.replace("¬", "").replace("†", "")
    return re.sub(r" +", " ", s).strip()

def run_generation(test_mode: bool):
    print("STARTING SCRIPT" + (" (TEST MODE)" if test_mode else ""))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1. Load model + tokenizer
    print("Loading GPT-2 model and tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        "ctrlg/gpt2-large_common-gen"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("ctrlg/gpt2-large_common-gen")

    # optional: clear CUDA cache
    torch.cuda.empty_cache()

    # 2. Load HMM
    print("Loading HMM with 4096 states")
    hmm = ctrlg.HMM.from_pretrained(
        "ctrlg/hmm_gpt2-large_common-gen_4096"
    ).to(device)
    print("HMM loaded")

    # 3. Build DFA for whole-word agentic + communal constraints
    print("Building DFA for agentic + communal constraints only")
    vs = hmm.vocab_size
    acb = ctrlg.AhoCorasickBuilder(vs)

    agentic = [" ambitious ", " assertive ", " bold ", " confident ", " decisive "]
    communal = [" caring ", " helpful ", " friendly ", " nurturing ", " supportive "]

    pats_a = [tokenizer.encode(w, add_special_tokens=False) for w in agentic]
    pats_c = [tokenizer.encode(w, add_special_tokens=False) for w in communal]

    prod = ctrlg.DFA_prod([acb.build(pats_a), acb.build(pats_c)], mode="intersection")
    dfa = ctrlg.DFAModel(prod, vs).to(device)
    print("DFA built successfully")

    # 4. Sampling parameters: enforce longer outputs
    MIN_TOK, MAX_TOK = 10, 20
    occupations = ["chef", "counselor", "writer", "scientist"]
    if test_mode:
        occupations = occupations[:2]
        target_per_prompt, batch_size = 3, 3
    else:
        target_per_prompt, batch_size = 500, 100

    outname = "output_test.csv" if test_mode else "output.csv"
    with open(outname, "w", newline="", encoding="utf8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["occupation", "sample", "label"])
        writer.writeheader()

        for idx, occ in enumerate(occupations, start=1):
            # seed prompt
            prompt_ids = tokenizer.encode(f"The {occ} was", add_special_tokens=False)
            print(f"Prompt {idx}/{len(occupations)}: The {occ} was")

            proc = ctrlg.ConstraintLogitsProcessor(
                hmm, dfa, MIN_TOK, MAX_TOK,
                prompt_ids=prompt_ids,
                prefix_ids=[],        # enforce only via DFA/HMM
                suffix_ids=[]
            )

            collected = 0
            batch_num = 0
            while collected < target_per_prompt:
                batch_num += 1
                bs = min(batch_size, target_per_prompt - collected)
                proc.hmm_batch_size = bs
                print(f"  Batch {batch_num}: sampling {bs} (collected {collected})")

                outputs = model.generate(
                    input_ids=torch.tensor([prompt_ids], device=device),
                    do_sample=True,
                    top_k=20,
                    top_p=0.8,
                    temperature=0.6,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    num_beams=5,
                    early_stopping=True,
                    min_new_tokens=MIN_TOK,
                    max_new_tokens=MAX_TOK,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([proc])
                )

                gens = ctrlg.extract_generated_ids(
                    outputs.tolist(),
                    prompt_ids,
                    suffix_ids=[],
                    eos_token_id=tokenizer.eos_token_id
                )
                for seq in gens:
                    sample = clean_text(tokenizer.decode(seq, skip_special_tokens=True))
                    if "_" in sample:
                        continue
                    writer.writerow({
                        "occupation": occ,
                        "sample":     sample,
                        "label":      "Ctrl-G GPT-2"
                    })
                    print("    →", sample)
                    collected += 1
                    if collected >= target_per_prompt:
                        break

            print(f"  Done {occ}: collected {collected} samples\n")

    print("ALL DONE — results in", outname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="small-scale test run (2 prompts, 3 samples each)"
    )
    args = parser.parse_args()
    run_generation(test_mode=args.test)
