from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("squad_v2")


def preprocess(example):
    inputs = tokenizer(
        example["question"],
        example["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        stride=128,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offset = inputs.pop("offset_mapping")
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}

    if len(example["answers"]["text"]) == 0:
        inputs["start_positions"] = torch.tensor(0)
        inputs["end_positions"] = torch.tensor(0)
    else:
        start_char = example["answers"]["answer_start"][0]
        end_char = start_char + len(example["answers"]["text"][0])
        sequence_ids = inputs["token_type_ids"]

        offsets = offset[0]
        start_token = end_token = 0
        for i, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_token = i
            if start < end_char <= end:
                end_token = i
        inputs["start_positions"] = torch.tensor(start_token)
        inputs["end_positions"] = torch.tensor(end_token)
    return inputs


def qa_collate_fn(batch):
    keys = batch[0].keys()
    output = {}
    for key in keys:
        # 判断是否是 tensor，否则跳过
        values = [item[key] for item in batch if isinstance(item[key], torch.Tensor)]
        if values and all(v.size() == values[0].size() for v in values):
            output[key] = torch.stack(values)
    return output


def get_train_data_loader():
    train_data = dataset["train"].map(preprocess)
    train_data.set_format(type="torch")
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=qa_collate_fn)
    return train_loader