import os
import warnings

import numpy as np
import torch
from hydra.experimental import compose, initialize
from transformers import (
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

import wandb
from dataloader import load

warnings.filterwarnings(action="ignore")


def main():
    initialize(config_path="./", strict=False)
    cfg = compose("config.yaml")

    tokenizer = AutoTokenizer.from_pretrained(cfg.PATH.tokenizer)
    # model = BartForConditionalGeneration.from_pretrained(
    #     "/home/nsw0311/nas_storage/models/DialogBart-finetuning"
    # ).cuda()
    model = BartForConditionalGeneration.from_pretrained(cfg.PATH.output_dir).cuda()

    user_token = "<user0>"
    model_token = "<user1>"
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    sep = tokenizer.sep_token_id

    talk_log = []

    while True:
        text_list = []
        while True:
            text = input("Me: ")
            if text == "":
                break
            elif text == "종료":
                exit()
            text_list.append(text)

        text = user_token + " ".join(text_list)
        talk_log.append(tokenizer.encode(text))

        cum_length = np.cumsum([len(i) for i in talk_log[::-1]])
        # cum_length = [3, 12, 15, 32, 61, 74, 85, 112, 123, 157, 179, 201]

        i = 0
        while cum_length[i] < model.config.max_position_embeddings - 2:
            i += 1
            if i >= len(talk_log):
                break

        input_ids = [bos]
        for j in range(-i, 0):
            input_ids += talk_log[j] + [sep]
        input_ids += [eos]

        input_ids = torch.tensor([input_ids]).cuda()
        decoder_input_ids = tokenizer.encode(model_token, return_tensors="pt").cuda()

        with torch.no_grad():
            pred = model.generate(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                max_length=model.config.max_position_embeddings,
                do_sample=True,
                early_stopping=True,
                top_k=10,
                bos_token_id=tokenizer.convert_tokens_to_ids(model_token),
                eos_token_id=eos,
                pad_token_id=tokenizer.pad_token_id,
            )[0, :-1].cpu()

        talk_log.append(pred.numpy().tolist())
        print("ai: " + tokenizer.decode(pred[1:]))


if __name__ == "__main__":
    main()
