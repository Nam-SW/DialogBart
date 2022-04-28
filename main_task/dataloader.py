from os.path import abspath, splitext
from typing import Literal, Optional, Union

import numpy as np
from datasets import load_dataset, logging
from torch.utils.data import Dataset

logging.set_verbosity(logging.ERROR)


def load(
    tokenizer,
    seq_len: int,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    shuffle_seed: Optional[int] = None,
    shuffle_after: bool = False,
):
    def _tokenize(data):
        tokenized = tokenizer(
            [w + c for w, c in zip(data["writer"], data["content"])],
            max_length=seq_len - 2,  # bos, eos
            truncation=True,
        )["input_ids"]

        return {
            "tokenized": tokenized,
            "talk_id": data["talk_id"],
        }

    def _grouping(data):
        input_ids = []
        decoder_input_ids = []

        queue = []
        response = []
        last_talk_id = data["talk_id"][0]

        for talk, ti in zip(data["tokenized"], data["talk_id"]):
            if last_talk_id != ti:
                queue = []
                response = []
                last_talk_id = ti

            if response:
                queue.append(response)
            response = talk

            if queue and response:
                ids = []
                for r in queue:
                    if ids:
                        ids.append(tokenizer.sep_token_id)
                    ids += r

                while len(ids) > seq_len - 2:
                    queue.pop(0)
                    ids = []
                    for r in queue:
                        if ids:
                            ids.append(tokenizer.sep_token_id)
                        ids += r

                input_ids.append(
                    [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
                )
                decoder_input_ids.append(response)

        e = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=seq_len,
            # return_tensors="pt",
        )
        d = tokenizer.pad(
            {"input_ids": decoder_input_ids},
            padding="max_length",
            max_length=seq_len,
            # return_tensors="pt",
        )
        l = tokenizer.pad(
            {
                "input_ids": [
                    dii[1:] + [tokenizer.eos_token_id] for dii in decoder_input_ids
                ]
            },
            padding="max_length",
            max_length=seq_len,
            # return_tensors="pt",
        )

        return {
            "input_ids": e["input_ids"],
            "attention_mask": e["attention_mask"],
            "decoder_input_ids": d["input_ids"],
            "decoder_attention_mask": d["attention_mask"],
            "labels": l["input_ids"],
        }

    train_data_path = abspath(train_data_path)
    is_eval = False
    _, extention = splitext(train_data_path)

    datafiles = {"train": train_data_path}
    if eval_data_path is not None:
        assert (
            train_test_split is None
        ), "Only one of eval_data_path and train_test_split must be entered."
        datafiles["test"] = abspath(eval_data_path)
        is_eval = True

    if train_test_split is not None:
        assert (
            0.0 < train_test_split < 1.0
        ), "train_test_split must be a value between 0 and 1"
        train_test_split = int(train_test_split * 100)
        train_test_split = {
            "train": f"train[:{train_test_split}%]",
            "test": f"train[{train_test_split}%:]",
        }
        is_eval = True

    data = load_dataset(
        extention.replace(".", ""),
        data_files=datafiles,
        split=train_test_split,
    )

    data = data.map(
        _tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )
    data = data.map(
        _grouping,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    if shuffle_seed and shuffle_after:
        data = data.shuffle(seed=shuffle_seed)

    return data["train"], (data["test"] if is_eval else None)
