import os
from transformers import (
    BertTokenizerFast, BertConfig, BertForMaskedLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import Dataset
from pathlib import Path

DATA_FILE = "wikipron_combined.tsv"
MODEL_DIR = "bert-ipa-model"
VOCAB_FILE = "ipa_vocab.txt"
MAX_LEN = 64

# Full list of IPA characters to use
IPA_CHARACTERS = ['l', 'ɖ', 'θ', 'œ', 'ʉ', 'ɳ', 'ʝ', 'ʐ', 'z', 'b', 'ʱ', 'o', 'ɒ', 'ɫ', 'y', 't', 'ŋ', 'ɟ', 'd', 'ɕ', 'x', 'ʑ', 'j', 'ɥ', 'ʕ', 'f', 'c', 'ð', 'ʁ', 'ɑ', 'ɜ', 'a', 'ɭ', 'm', 'ʲ', 'v', 'h', 'ɡ', 'ʈ', 'ʔ', 'ɔ', '͡', 'k', 'ʂ', 'q', 'ə', 'χ', 'ʒ', 'ɾ', 'n', 'w', 'ɛ', 'p', 'ʏ', 'ɦ', 'ʃ', 'ɤ', 'ɪ', 'ʋ', 'ɐ', 'ɣ', 'ɽ', 'ɯ', 'ɲ', 'u', 'i', 'ʊ', 'ħ', 'ʀ', 's', 'e', 'ɨ', 'ø', 'r', 'ː', 'β', 'æ']

# Create vocab file for tokenizer
def write_vocab_file(characters, vocab_path):
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token in special_tokens + characters:
            f.write(token + "\n")

def load_ipa_data():
    lines = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                _, ipa = parts
                char_seq = " ".join(list(ipa))  # space-separated characters
                lines.append({"text": char_seq})
    return Dataset.from_list(lines)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    write_vocab_file(IPA_CHARACTERS, VOCAB_FILE)

    tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE, do_lower_case=False, tokenize_chinese_chars=False)
    
    dataset = load_ipa_data()
    print(f"✅ Loaded {len(dataset)} IPA sequences.")

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    tokenized_ds = dataset.map(tokenize, batched=True)

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=MAX_LEN,
    )

    model = BertForMaskedLM(config)

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=128,
        num_train_epochs=2,
        logging_dir=None,  # disables TensorBoard
        report_to=[], 
        logging_steps=1000,
        save_total_limit=2,
        overwrite_output_dir=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    )

    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"✅ Trained model and tokenizer saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()