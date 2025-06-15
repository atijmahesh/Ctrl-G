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
    # Remove NBSPs and stray chars, collapse spaces
    s = raw.replace("\u00A0", " ")
    s = s.replace("¬", "").replace("†", "")
    return re.sub(r" +", " ", s).strip()

def run_generation(test_mode: bool):
    print("STARTING SCRIPT" + (" (TEST MODE)" if test_mode else ""))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Load GPT-2 + tokenizer
    print("Loading GPT-2 model and tokenizer")
    model = AutoModelForCausalLM.from_pretrained("ctrlg/gpt2-large_common-gen")\
             .to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("ctrlg/gpt2-large_common-gen")

    # 2) Load HMM
    print("Loading HMM with 4096 states")
    hmm = ctrlg.HMM.from_pretrained("ctrlg/hmm_gpt2-large_common-gen_4096")\
             .to(device)
    print("HMM loaded")

    # 3) Build the keyphrase DFA once
    print("Building DFA for agentic + communal constraints only")
    vs = hmm.vocab_size
    acb = ctrlg.AhoCorasickBuilder(vs)

    agentic = [
        " ambitious ", " assertive ", " bold ",
        " confident ", " decisive ",
        " independent ", " self-reliant ",
        " competitive ", " adventurous "
    ]
    communal = [
        " accommodating ", " caring ", " cooperative ",
        " empathetic ", " friendly ",
        " nurturing ", " supportive ",
        " compassionate ", " helpful ", " loyal "
    ]
    pats_a = [tokenizer.encode(w, add_special_tokens=False) for w in agentic]
    pats_c = [tokenizer.encode(w, add_special_tokens=False) for w in communal]
    prod  = ctrlg.DFA_prod([acb.build(pats_a), acb.build(pats_c)],
                            mode="intersection")
    dfa   = ctrlg.DFAModel(prod, vs).to(device)
    print("DFA built successfully")

    # 4) Setup sampling bounds & occupations
    MIN_TOK, MAX_TOK = 10, 20
    occupations = [
        "The architect was", "The artist was", "The barista was",
        "The chef was",     "The counselor was", "The doctor was",
        "The electrician was","The engineer was",  "The journalist was",
        "The lawyer was",   "The mechanic was",  "The nurse was",
        "The pharmacist was","The photographer was","The pilot was",
        "The plumber was",  "The scientist was", "The teacher was",
        "The salesperson was","The writer was"
    ]
    if test_mode:
        occupations = occupations[:2]
        target_per_prompt, batch_size = 3, 50
    else:
        target_per_prompt, batch_size = 500, 100

    # prepare the period token for suffix & eos
    period_id = tokenizer.encode(".", add_special_tokens=False)[0]

    outname = "output_test.csv" if test_mode else "output.csv"
    with open(outname, "w", newline="", encoding="utf8") as fout:
        writer = csv.DictWriter(fout,
                                fieldnames=["occupation", "sample", "label"])
        writer.writeheader()

        for idx, prompt in enumerate(occupations, start=1):
            print(f"Prompt {idx}/{len(occupations)}: {prompt}")
            prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)

            seen = set()
            collected = 0
            batch_num = 0

            while collected < target_per_prompt:
                batch_num += 1
                bs = min(batch_size, target_per_prompt - collected)

                # 5) Re-create the processor fresh each batch
                proc = ctrlg.ConstraintLogitsProcessor(
                    hmm, dfa, MIN_TOK, MAX_TOK,
                    prompt_ids=prefix_ids,
                    prefix_ids=[],           # only DFA controls prefixes
                    suffix_ids=[period_id]   # stop once "." appears
                )
                proc.hmm_batch_size = bs

                print(f"  Batch {batch_num}: sampling {bs} (collected {collected})")
                outputs = model.generate(
                    input_ids=torch.tensor([prefix_ids], device=device),
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.0,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    num_beams=1,
                    min_new_tokens=MIN_TOK,
                    max_new_tokens=MAX_TOK,
                    eos_token_id=period_id,        # halt on period
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([proc])
                )

                gens = ctrlg.extract_generated_ids(
                    outputs.tolist(),
                    prefix_ids,
                    suffix_ids=[period_id],
                    eos_token_id=tokenizer.eos_token_id
                )
                for seq in gens:
                    sample = clean_text(
                        tokenizer.decode(seq, skip_special_tokens=True)
                    )
                    if "_" in sample or sample in seen:
                        continue
                    seen.add(sample)
                    writer.writerow({
                        "occupation": prompt,
                        "sample":     sample,
                        "label":      "Ctrl-G GPT-2"
                    })
                    collected += 1
                    # only log every 100th unique sample
                    if test_mode or collected % 100 == 0:
                        print("    →", sample)
                    if collected >= target_per_prompt:
                        break

            print(f"  Done {prompt}: collected {collected} samples\n")

    print("ALL DONE — results in", outname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="small-scale test run (2 prompts, 50 samples each)")
    args = parser.parse_args()
    run_generation(test_mode=args.test)
