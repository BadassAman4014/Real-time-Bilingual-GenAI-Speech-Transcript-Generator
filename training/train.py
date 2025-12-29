import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Trainer, TrainingArguments
from dataset import STTDataset

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

dataset = STTDataset("data/manifests/train.json")

def collate(batch):
    audio = [x["audio"] for x in batch]
    text = [x["text"] for x in batch]

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    labels = processor.tokenizer(
        text,
        return_tensors="pt",
        padding=True
    ).input_ids

    inputs["labels"] = labels
    return inputs

args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    fp16=torch.cuda.is_available(),
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate
)

trainer.train()
