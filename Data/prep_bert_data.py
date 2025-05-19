import os
import csv
import unicodedata

# === Normalization map ===
symbol_map = {
    "g": "ɡ", "ʤ": "d͡ʒ", "ʧ": "t͡ʃ", "ʣ": "d͡z", "ʦ": "t͡s",
    "tʃ": "t͡ʃ", "dʒ": "d͡ʒ", "ts": "t͡s", "dz": "d͡z", "tɕ": "t͡ɕ", "dʑ": "d͡ʑ",
    ":": "ː", "ˑ": "ː", "ːː": "ː",
    "ˈ": "", "ˌ": "", "˥": "", "˦": "", "˧": "", "˨": "", "˩": "",
    "˦˥": "", "˨˦": "", "˧˨": "",
    " ": "", "'": "", "’": "", "ʼ": "", "‿": "", "-": "",
    "(": "", ")": "", ".": "", "[": "", "]": "", "#": "", ";": "",
    "ˀ": "", "ˤ": "", "ʰ": "", "ⁿ": "", "ᵐ": "", "ⁿ̥": "",
    "̹": "", "̜": "", "̃": "", "̥": "", "̊": "", "̬": "",
    "̄": "", "̆": "", "¹": "", "²": "", "³": "", "⁴": "", "⁵": "",
    "ʳ": "", "ʴ": "", "ʶ": "", "˞": "", "̯": "",
    "ə̆": "ə", "ə̃": "ə", "ə̯": "ə"
}

# === Normalize and filter ===
def normalize_ipa(text):
    text = unicodedata.normalize("NFC", text)
    for k, v in symbol_map.items():
        text = text.replace(k, v)
    return text

def filter_ipa(text, valid_symbols):
    return "".join([c for c in normalize_ipa(text) if c in valid_symbols])

# === Collect IPA symbols ===
def collect_wikipron_symbols(directory):
    symbols = set()
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                ipa = normalize_ipa(row[-1].replace(" ", ""))
                symbols.update(ipa)
    return symbols

def collect_corpus_symbols(paths):
    symbols = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ipa = normalize_ipa(row["transcription"])
                symbols.update(ipa)
    return symbols

# === Paths ===
wikipron_dir = "filtered_wikipron_scrapes"
own_files = ["corpus_all_train.csv", "corpus_all_test.csv"]

# === Compute intersection ===
wikipron_set = collect_wikipron_symbols(wikipron_dir)
corpus_set = collect_corpus_symbols(own_files)
intersect = wikipron_set & corpus_set

print(f"✅ Found {len(intersect)} shared IPA symbols")

# === Write cleaned wikipron file ===
with open("wikipron_combined.tsv", "w", encoding="utf-8", newline="") as out_f:
    writer = csv.writer(out_f, delimiter="\t")
    for filename in os.listdir(wikipron_dir):
        with open(os.path.join(wikipron_dir, filename), encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                ortho, raw_ipa = row[0], row[-1]
                cleaned_ipa = filter_ipa(raw_ipa.replace(" ", ""), intersect)
                if cleaned_ipa:
                    writer.writerow([ortho, cleaned_ipa])

print("📄 Saved: wikipron_combined.tsv")

# === Clean and write own corpora ===
def clean_corpus(infile, outfile):
    with open(infile, encoding="utf-8") as f_in, open(outfile, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            cleaned = filter_ipa(row["transcription"], intersect)
            if cleaned:
                row["transcription"] = cleaned
                writer.writerow(row)

clean_corpus("corpus_all_train.csv", "corpus_clean_train.csv")
clean_corpus("corpus_all_test.csv", "corpus_clean_test.csv")

print("📄 Saved: corpus_clean_train.csv, corpus_clean_test.csv")
print(intersect)
print(len(intersect))