from os.path import abspath, splitext
from typing import Literal, Optional, Union

import numpy as np
from datasets import load_dataset, logging
from torch.utils.data import Dataset

logging.set_verbosity(logging.ERROR)


class BartTextNoiseDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        seq_len: int,
        mask_ratio: float = 0.15,
        random_ratio: float = 0.2,
        insert_ratio: float = 0.15,
        rotate_ratio: float = 0.2,
        permute_sentence_ratio: float = 0.2,
        replace_length: Literal[-1, 0, 1] = 1,
        mask_length: Literal["subword", "word", "span-poisson"] = "span-possion",
        random_seed: int = 42,
    ):
        self.data = data
        self.tokenizer = tokenizer

        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.insert_ratio = insert_ratio
        self.rotate_ratio = rotate_ratio
        self.permute_sentence_ratio = permute_sentence_ratio

        self.replace_length = replace_length
        self.mask_length = mask_length

        self.set_parameters()
        np.random.seed(random_seed)

    def set_parameters(self):
        ps = None
        if self.mask_length == "span-possion":
            _lambda = 3.0  # poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = np.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break

        self.ps = ps

        self.full_stop_tokens = [
            self.tokenizer.pad_token_id,
            self.tokenizer.convert_tokens_to_ids("<user0>"),
            self.tokenizer.convert_tokens_to_ids("<user1>"),
        ]

    def mask_span_sample(self, n):
        return np.random.choice(len(self.ps), n, p=self.ps)

    def permute_sentences(self, source, p=1.0):
        full_stops = np.isin(source, self.full_stop_tokens)

        # Tokens that are full stops, where the previous token is not
        sentence_ends = np.nonzero(full_stops[1:] * ~full_stops[:-1])[0] + 1
        result = source.copy()

        num_sentences = sentence_ends.size
        num_to_permute = np.ceil((num_sentences * 2 * p) / 2.0).astype(int)
        substitutions = np.random.permutation(num_sentences)[:num_to_permute]
        ordering = np.arange(0, num_sentences)
        ordering[substitutions] = substitutions[np.random.permutation(num_to_permute)]

        index = 0
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 0) : sentence_ends[i]]
            result[index : index + sentence.size] = sentence
            index += sentence.size
        return result

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(np.ceil(num_tokens * p))

        noise_indices = np.random.permutation(num_tokens + n - 2)[:n] + 1
        noise_mask = np.zeros((num_tokens + n,), dtype=np.bool)
        noise_mask[noise_indices] = 1
        result = np.full(n + len(tokens), -1)

        num_random = int(np.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.tokenizer.mask_token_id
        result[noise_indices[:num_random]] = np.random.randint(
            low=1, high=self.tokenizer.vocab_size, size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def word_starts(self, source):
        special_tokens = self.tokenizer.special_tokens_map["additional_special_tokens"]
        special_tokens = [t for t in special_tokens if t not in ["<name>", "<url>"]]
        special_tokens = self.tokenizer.convert_tokens_to_ids(special_tokens)

        return ~np.isin(source, special_tokens)

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)  # Boolean D1 tensor.
        num_to_mask = int(np.ceil(is_word_start.sum() * p))
        num_inserts = 0

        # 마스킹 안할거면 NAGA
        if num_to_mask == 0:
            return source

        # 정의된 확률분포가 있다면
        if self.ps is not None:
            # cum_sum 을 왜 사용하는지 모르겠다
            # -> 마스킹 위치별 개수를 나타내는건가
            # --> 그럴듯해
            lengths = self.mask_span_sample(num_to_mask)
            cum_length = np.cumsum(lengths, 0)

            # 너무 짧으면 개수 맞을때까지 더하고
            while cum_length[-1] < num_to_mask:
                lengths = np.concatenate(
                    [
                        lengths,
                        self.mask_span_sample(num_to_mask),
                    ],
                    axis=0,
                )
                cum_length = np.cumsum(lengths, 0)

            # 길이 맞을때까지의 인덱스만 가져와서
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            # 마지막 길이는 남으면 길이 맞게 설정
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            # 거기까지만 사용
            lengths = lengths[:num_to_mask]

            # insert 가 아니라 masking하는것들
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size
            num_to_mask -= num_inserts

            # insert 밖에 없으면 걍 그거만 하고 넘기기
            # 근데 이런 경우가 나오나? 예외처리 할 필요가 있음?
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size)

            assert (lengths > 0).all()

        # 확률분포 없으면 걍 기본적인 마스킹
        else:
            lengths = np.ones((num_to_mask,), dtype=np.float32)

        # special token은 제외
        word_starts_idx = np.array(is_word_start.nonzero()[0])
        # random으로 뽑아서 가져옴. -> 걍 sample 하면 되지 않나
        indices = word_starts_idx[
            np.random.permutation(word_starts_idx.size)[:num_to_mask]
        ]

        # 임계값 맞춰서 masking이 아닌 replace idx 가져오기
        mask_random = np.random.random_sample(num_to_mask) < self.random_ratio

        source_length = source.size
        # eos 챙기는듯 -> bos는?
        assert source_length - 1 not in indices
        to_keep = np.ones(source_length, dtype=np.bool)
        is_word_start[-1] = 255

        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # 일단 다 마스킹하고
            source[indices] = self.tokenizer.mask_token_id
            # replace
            source[indices[mask_random]] = np.random.randint(
                1, self.tokenizer.vocab_size, size=(mask_random.sum(),)
            )

        if self.ps is not None:
            assert lengths.ndim == 1
            assert lengths.size == indices.size
            # lengths -= 1

        while indices.size > 0:
            if self.ps is not None:
                assert lengths.size == indices.size

            completed_idx = is_word_start[indices]

            if self.replace_length != -1:
                # delete token
                to_keep[indices[completed_idx]] = 0
            else:
                # keep index, but replace it with [MASK]
                temp_idx = mask_random[completed_idx]
                source[temp_idx] = self.tokenizer.mask_token_id
                source[temp_idx[mask_random[completed_idx]]] = np.random.randint(
                    1,
                    self.tokenizer.vocab_size,
                    size=(mask_random[completed_idx].sum(),),
                )

            lengths[is_word_start[indices]] -= 1
            lengths[indices + 1 >= source.size] -= 1
            indices[indices + 1 < source.size] += 1

            indices = indices[lengths > 0]
            lengths = lengths[lengths > 0]

            # if self.ps is not None:
            #     assert lengths.size == indices.size
            #     lengths -= is_word_start[indices]
            #     uncompleted = lengths >= 0
            #     completed = lengths <= 0
            # else:
            #     uncompleted = is_word_start[indices] == 0
            #     completed = is_word_start[indices] != 0

            # mask_indices = indices[completed]
            # now_mask_random = mask_random[completed]

            # indices = indices[uncompleted] + 1
            # mask_random = mask_random[uncompleted]

            # usable_idx = indices < source.size
            # indices = indices[usable_idx]
            # mask_random = mask_random[usable_idx]

            # if self.ps is not None:
            #     lengths = lengths[uncompleted]
            #     lengths = lengths[usable_idx]

            # if self.replace_length != -1:
            #     # delete token
            #     to_keep[mask_indices] = 0
            # else:
            #     # keep index, but replace it with [MASK]
            #     source[mask_indices] = self.tokenizer.mask_token_id
            #     source[mask_indices[now_mask_random]] = np.random.randint(
            #         1, self.tokenizer.vocab_size, size=(now_mask_random.sum(),)
            #     )

            # if self.ps is None:
            #     assert source_length - 1 not in indices

        source = source[to_keep]
        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size)

        return source

    def add_rolling_noise(self, tokens):
        offset = np.random.randint(1, max(1, tokens.size - 1) + 1)
        tokens = np.concatenate(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            axis=0,
        )
        return tokens

    def _pad(self, data):
        input_ids = {"input_ids": data["input_ids"]}
        decoder_input_ids = {"input_ids": data["decoder_input_ids"]}
        labels = {"input_ids": data["labels"]}

        e = self.tokenizer.pad(
            input_ids,
            padding="max_length",
            max_length=512,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        d = self.tokenizer.pad(
            decoder_input_ids,
            padding="max_length",
            max_length=512,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        l = self.tokenizer.pad(
            labels,
            padding="max_length",
            max_length=512,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

        return {
            "input_ids": e["input_ids"],
            "attention_mask": e["attention_mask"],
            "decoder_input_ids": d["input_ids"],
            "decoder_attention_mask": d["attention_mask"],
            "labels": l["input_ids"],
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Union[int, slice]):
        sample = self.data[idx]
        if isinstance(idx, int):
            sample = {k: [v] for k, v in sample.items()}
        input_ids = sample["input_ids"]

        result = []
        for corrupted_input in input_ids:
            corrupted_input = np.array(corrupted_input)

            if self.permute_sentence_ratio > 0.0:
                corrupted_input = self.permute_sentences(
                    corrupted_input, self.permute_sentence_ratio
                )

            if self.mask_ratio > 0:
                corrupted_input = self.add_whole_word_mask(
                    corrupted_input, self.mask_ratio
                )

            if self.insert_ratio > 0:
                corrupted_input = self.add_insertion_noise(
                    corrupted_input, self.insert_ratio
                )

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                corrupted_input = self.add_rolling_noise(corrupted_input)

            result.append(corrupted_input[: self.seq_len].tolist())

        sample["input_ids"] = result
        sample = self._pad(sample)

        if isinstance(idx, int):
            sample = {k: v[0] for k, v in sample.items()}

        return sample


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
    mask_ratio: float = 0.15,
    random_ratio: float = 0.2,
    insert_ratio: float = 0.15,
    rotate_ratio: float = 0.2,
    permute_sentence_ratio: float = 0.2,
    replace_length: Literal[-1, 0, 1] = 1,
    mask_length: Literal["subword", "word", "span-poisson"] = "span-possion",
):
    def _tokenize(data):
        data["tokenized"] = tokenizer.encode(
            data["writer"] + data["content"],
            max_length=seq_len - 2,  # bos, eos
            truncation=True,
        )

        return data

    def _grouping(data):
        input_ids = []

        last_input_ids = [tokenizer.bos_token_id]
        last_writer = data["writer"][0]
        last_talk_id = data["talk_id"][0]

        for ii, wr, ti in zip(data["tokenized"], data["writer"], data["talk_id"]):
            if (
                # last_writer == wr
                # and
                len(last_input_ids + ii) <= seq_len - 1
                and last_talk_id == ti
            ):
                last_input_ids += ii

            else:
                last_input_ids += [tokenizer.eos_token_id]
                input_ids.append(last_input_ids)

                last_input_ids = [tokenizer.bos_token_id] + ii
                last_talk_id = ti

            # last_writer = wr

        tokenized = {
            "input_ids": input_ids,
            "decoder_input_ids": [i[:-1] for i in input_ids],
            "labels": [i[1:] for i in input_ids],
        }
        return tokenized

        # pad_tokenized = tokenizer.pad(
        #     tokenized, max_length=seq_len, padding="max_length"
        # )

        # return pad_tokenized

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
        num_proc=worker,
    )
    data = data.map(
        _grouping,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    if shuffle_seed and not shuffle_after:
        data = data.shuffle(seed=shuffle_seed)

    train_data = BartTextNoiseDataset(
        data["train"],
        tokenizer,
        seq_len=seq_len,
        mask_ratio=mask_ratio,
        random_ratio=random_ratio,
        insert_ratio=insert_ratio,
        rotate_ratio=rotate_ratio,
        permute_sentence_ratio=permute_sentence_ratio,
        replace_length=replace_length,
        mask_length=mask_length,
    )
    eval_data = (
        BartTextNoiseDataset(
            data["test"],
            tokenizer,
            seq_len=seq_len,
            mask_ratio=mask_ratio,
            random_ratio=random_ratio,
            insert_ratio=insert_ratio,
            rotate_ratio=rotate_ratio,
            permute_sentence_ratio=permute_sentence_ratio,
            replace_length=replace_length,
            mask_length=mask_length,
        )
        if is_eval
        else None
    )

    if shuffle_seed and shuffle_after:
        data = data.shuffle(seed=shuffle_seed)

    return train_data, eval_data
