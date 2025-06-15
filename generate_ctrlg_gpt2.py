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

    # 2. Clear cache
    torch.cuda.empty_cache()

    # 3. Load HMM
    print("Loading HMM with 4096 states")
    hmm = ctrlg.HMM.from_pretrained(
        "ctrlg/hmm_gpt2-large_common-gen_4096"
    ).to(device)
    print("HMM loaded")

    # 4. Build DFA using past style (agentic + communal + word-count)
    print("Building DFA with agentic, communal, and word-count constraints")
    vocab_size   = hmm.vocab_size
    eos_token_id = hmm.eos_token_id

    acb = ctrlg.AhoCorasickBuilder(vocab_size)
    wcb = ctrlg.WordCountBuilder(tokenizer, vocab_size)

    agentic = [" ambitious", " assertive", " bold", " confident", " decisive"]
    communal = [" unselfish", " accommodating", " caring",
                " cooperative", " empathetic", " friendly"]

    dfa_graphs = []
    for kp in (agentic, communal):
        pats = [tokenizer.encode(x, add_special_tokens=False) for x in kp]
        dfa_graphs.append(acb.build(pats))

    # exactly 8-15 words generated
    dfa_graphs.append(wcb.build(8, 15))

    prod = ctrlg.DFA_prod(dfa_graphs, mode="intersection")
    dfa  = ctrlg.DFAModel(prod, vocab_size).to(device)
    print("DFA built")

    # 5. Prepare suffix to stop on period + end-of-text
    suffix = ".<|endoftext|>"
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    # 6. Generation bounds
    MIN_TOK, MAX_TOK = 5, 32
    occupations = [
        "The chef was",
        "The counselor was",
        "The writer was",
        "The scientist was"
    ]
    if test_mode:
        occupations = occupations[:2]
        target_per_prompt, batch_size = 3, 3
    else:
        target_per_prompt, batch_size = 500, 100

    outname = "output_test.csv" if test_mode else "output.csv"
    with open(outname, "w", newline="", encoding="utf8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["occupation", "sample", "label"])
        writer.writeheader()

        for idx, prompt in enumerate(occupations, start=1):
            print(f"Prompt {idx}/{len(occupations)}: {prompt}")
            prefix_ids = tokenizer.encode(prompt + " ", add_special_tokens=False)

            proc = ctrlg.ConstraintLogitsProcessor(
                hmm, dfa, MIN_TOK, MAX_TOK,
                prompt_ids=prefix_ids,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids
            )

            collected = 0
            batch_num = 0
            while collected < target_per_prompt:
                batch_num += 1
                bs = min(batch_size, target_per_prompt - collected)
                proc.hmm_batch_size = bs
                print(f"  Batch {batch_num}: sampling {bs} (collected {collected})")

                outputs = model.generate(
                    input_ids=torch.tensor([prefix_ids], device=device),
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
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
                    suffix_ids=suffix_ids,
                    eos_token_id=tokenizer.eos_token_id
                )
                for seq in gens:
                    sample = clean_text(tokenizer.decode(seq, skip_special_tokens=True))
                    if "_" in sample:
                        continue
                    writer.writerow({
                        "occupation": prompt,
                        "sample":     sample,
                        "label":      "Ctrl-G GPT-2"
                    })
                    print("    →", sample)
                    collected += 1
                    if collected >= target_per_prompt:
                        break

            print(f"  Done {prompt}: collected {collected} samples\n")

    print("ALL DONE — results in", outname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="small-scale test run (2 prompts, 3 samples each)")
    args = parser.parse_args()
    run_generation(test_mode=args.test)
