#!/usr/bin/env python3
import os

os.environ["CUDA_VISIBLE_DEVICES"]    = "0"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["TORCHINDUCTOR_DISABLE"]   = "1"
os.environ["INDUCTOR_DISABLE_TRITON"] = "1"

import torch
import ctrlg
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList


def main():
    # -- Step 3.1: Load model + HMM ------------------------------------------
    print("[1/5] Setting up device and environment...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"       * Device: {device}")

    BASE_MODEL_PATH = "ctrlg/gpt2-large_common-gen"
    HMM_MODEL_PATH  = "ctrlg/hmm_gpt2-large_common-gen_4096"

    print(f"[2/5] Loading GPT-2 checkpoint from {BASE_MODEL_PATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device).eval()
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    print(f"[3/5] Loading HMM from {HMM_MODEL_PATH}...")
    hmm_model = ctrlg.HMM.from_pretrained(HMM_MODEL_PATH).to(device)

    # -- Step 1.2: Define your prompts & synonyms ---------------------------
    occupations = [
        "The architect was",
        "The artist was",
        "The barista was",
        "The chef was",
        "The counselor was",
        "The doctor was",
        "The electrician was",
        "The engineer was",
        "The journalist was",
        "The lawyer was",
        "The mechanic was",
        "The nurse was",
        "The pharmacist was",
        "The photographer was",
        "The pilot was",
        "The plumber was",
        "The scientist was",
        "The teacher was",
        "The salesperson was",
        "The writer was",
    ]

    agentic = [
        " ambitious ", " assertive ", " bold ", " confident ", " decisive ",
        " independent ", " self-reliant ", " competitive ", " adventurous ", " assertive "
    ]
    communal = [
        " accommodating ", " caring ", " cooperative ", " empathetic ", " friendly ",
        " nurturing ", " supportive ", " compassionate ", " helpful ", " loyal "
    ]

    # -- Step 3.2: Build DFA constraint once and reuse ---------------------
    print(f"[4/5] Building DFA constraint (>=1 agentic + >=1 communal, 8-15 words)...")
    vocab_size = hmm_model.vocab_size

    ac_builder = ctrlg.AhoCorasickBuilder(vocab_size)
    wc_builder = ctrlg.WordCountBuilder(tokenizer, vocab_size)

    # encode keyphrases
    agentic_pats = [tokenizer.encode(x, add_special_tokens=False) for x in agentic]
    communal_pats = [tokenizer.encode(x, add_special_tokens=False) for x in communal]

    dfa_graphs = [
        ac_builder.build(agentic_pats),
        ac_builder.build(communal_pats),
        wc_builder.build(8, 15)
    ]
    dfa_prod = ctrlg.DFA_prod(dfa_graphs, mode="intersection")
    dfa_model = ctrlg.DFAModel(dfa_prod, vocab_size).to(device)

    # -- Step 3.3 & 3.4: Sampling + write CSV -------------------------------
    OUTPUT_CSV = "ctrlg_gpt2_step3.csv"
    print(f"[5/5] Generating samples and writing to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=["occupation","sample","label"])
        writer.writeheader()

        for prompt in occupations:
            prompt_text = prompt + " "
            print(f"Prompt: \"{prompt_text}\"")
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            min_new_tokens = 8
            max_new_tokens = 15

            constraint_proc = ctrlg.ConstraintLogitsProcessor(
                hmm_model,
                dfa_model,
                min_new_tokens,
                max_new_tokens,
                prompt_ids,
                prefix_ids=[],
                suffix_ids=[]
            )
            constraint_proc.hmm_batch_size = 8

            # we'll sample in batches of 100 to avoid OOM
            total = 500
            batch_size = 100
            collected = 0

            while collected < total:
                bs = min(batch_size, total - collected)
                print(f"sampling batch {collected+1}-{collected+bs}...")
                outputs = base_model.generate(
                    input_ids=torch.tensor([prompt_ids], device=device),
                    do_sample=True,
                    top_p=0.95,
                    temperature=1.0,
                    num_beams=1,
                    num_return_sequences=bs,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([constraint_proc]),
                )

                # strip off the prompt and convert to text
                gen_ids = ctrlg.extract_generated_ids(
                    outputs.tolist(),
                    prompt_ids,
                    suffix_ids=[],
                    eos_token_id=tokenizer.eos_token_id
                )
                for seq in gen_ids:
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    writer.writerow({
                        "occupation": prompt,
                        "sample": text,
                        "label": "Ctrl-G GPT-2"
                    })

                collected += bs

    print("All done!")

if __name__ == "__main__":
    main()
