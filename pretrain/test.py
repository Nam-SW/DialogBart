import datetime as dt
import random
import re

import hydra
import numpy as np
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def processing(text: str) -> str:
    result = re.sub(r"[^.,:\(\)\[\]%\- 가-힣A-Za-z0-9]", " ", str(text))
    result = result.strip()
    return result


@hydra.main(config_name="config.yaml")
def main(cfg):
    model = BartForConditionalGeneration.from_pretrained(cfg.TEST.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.TEST.tokenizer)

    now = dt.datetime.now()

    with open(cfg.TEST.test_set, "r", encoding="utf-8") as f:
        for line in f.readlines():
            text = processing(line)
            print(line + "\n")
            input_data = tokenizer(
                text,
                return_tensors="pt",
                max_length=cfg.DATASETS.seq_len,
                padding="max_length",
                truncation=True,
            )

            set_seed(cfg.TEST.random_seed)

            pred = model.generate(
                **input_data,
                max_length=cfg.DATASETS.seq_len,
                do_sample=True,
                top_k=20,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ).numpy()

            for s in pred:
                print()
                s = s[s != tokenizer.pad_token_id]
                s = s[1:-1]
                text_list = list(
                    set([i.strip() for i in tokenizer.decode(s).split("<ssep>")])
                )
                for t in text_list:
                    print(t)

            print("\n" + "-" * 100 + "\n")

    print((dt.datetime.now() - now).total_seconds())


if __name__ == "__main__":
    main()
