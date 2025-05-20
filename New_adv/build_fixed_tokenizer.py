from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import pandas as pd

# === Load IPA word list from your training data ===
df = pd.read_csv("../Data/corpus_clean_train.csv")
ipa_words = df['transcription'].tolist()

# === Setup character-level tokenizer ===
pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
tokenizer_model = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer_model.pre_tokenizer = pre_tokenizer

trainer = trainers.WordLevelTrainer(
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer_model.train_from_iterator(ipa_words, trainer)

# === Wrap it as a HuggingFace tokenizer ===
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_model,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# === Save it to disk ===
wrapped_tokenizer.save_pretrained("fixed_ipa_tokenizer")
print("✅ Saved fixed tokenizer to 'fixed_ipa_tokenizer/'")
