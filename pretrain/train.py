import os
import warnings

# import hydra
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

    # tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(cfg.PATH.tokenizer)
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    # 모델 로드
    model = BartForConditionalGeneration(
        BartConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
            **cfg.MODEL,
        )
    )

    # if cfg.ETC.get("wandb_project"):
    #     os.environ["WANDB_PROJECT"] = cfg.ETC.wandb_project

    # 학습 arg 세팅
    args = TrainingArguments(
        do_train=True,
        do_eval=eval_dataset is not None,
        logging_dir=cfg.PATH.logging_dir,
        output_dir=cfg.PATH.checkpoint_dir,
        **cfg.TRAININGARGS,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=None,
        data_collator=default_data_collator,
    )

    # 학습 시작
    trainer.train()
    # trainer.train(resume_from_checkpoint=cfg.ETC.get("load_checkpoint"))

    # # 모델 저장
    trainer.save_model(cfg.PATH.output_dir)

    # if cfg.ETC.get("wandb_project"):
    #     wandb.finish()


if __name__ == "__main__":
    main()
