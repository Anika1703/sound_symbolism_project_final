from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-ipa-model")
sample = "sɑ̃kʁi"  # or whatever IPA word you want
encoding = tokenizer(sample, return_tensors="pt")

print("Tokens:", tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))
print("Input IDs:", encoding['input_ids'][0])
print("Vocab size:", len(tokenizer.get_vocab()))
