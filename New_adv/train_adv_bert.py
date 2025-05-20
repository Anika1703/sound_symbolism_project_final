import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import BertModel, PreTrainedTokenizerFast
from torch.optim.lr_scheduler import StepLR

def parse_args():
    parser = argparse.ArgumentParser(description='Train Adversarial Sound Symbolism Model with BERT')
    parser.add_argument('--bert_model_name_or_path', type=str, required=True,
                        help='Path to pretrained BERT (e.g., bert-ipa-model)')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to train CSV (with columns transcription,label,lang_fam)')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test CSV')
    parser.add_argument('--output', type=str, default='adv_model',
                        help='Directory to save the fine-tuned model')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_length', type=int, default=64, help='Max sequence length')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of projected embeddings')
    parser.add_argument('--lambda_param', type=float, default=0.05,
                        help='Weight for adversarial loss')
    return parser.parse_args()


class IPADataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df['transcription'].tolist()
        self.symbolism_labels = torch.tensor(df['label'].values, dtype=torch.long)
        self.language_labels = torch.tensor(df['lang_fam'].values, dtype=torch.long)
        self.encodings = tokenizer(
            self.texts,
            padding='max_length', truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.symbolism_labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        return item, self.symbolism_labels[idx], self.language_labels[idx]


class AdversarialSoundSymbolismModel(nn.Module):
    def __init__(self, bert_model_name_or_path: str, embedding_dim: int, num_languages: int, lambda_param: float):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name_or_path)
        hidden_size = self.bert.config.hidden_size
        self.embedding_layer = nn.Linear(hidden_size, embedding_dim)
        self.dropout = nn.Dropout(0.4)
        self.size_classifier = nn.Linear(embedding_dim, 2)
        self.language_classifier = nn.Linear(embedding_dim, num_languages)

        self.lambda_param = lambda_param

        self.encoder_optimizer = optim.Adam(
            list(self.bert.parameters()) +
            list(self.embedding_layer.parameters()) +
            list(self.size_classifier.parameters()),
            lr=2e-5, weight_decay=1e-4
        )
        self.language_optimizer = optim.Adam(
            self.language_classifier.parameters(), lr=1e-4
        )
        self.symbolism_loss_fn = nn.CrossEntropyLoss()
        self.language_loss_fn = nn.CrossEntropyLoss()

    def forward_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        embeddings = self.dropout(self.embedding_layer(pooled))
        return embeddings

    def train_language_classifier(self, inputs, language_labels):
        self.bert.eval()
        self.embedding_layer.eval()
        self.size_classifier.eval()
        self.language_classifier.train()

        self.language_optimizer.zero_grad()
        embeddings = self.forward_embeddings(**inputs).detach()
        lang_logits = self.language_classifier(self.dropout(embeddings))
        loss = self.language_loss_fn(lang_logits, language_labels)
        loss.backward()
        self.language_optimizer.step()
        return loss.item()

    def train_encoder_symbolism(self, inputs, symbolism_labels, language_labels, current_lambda):
        self.bert.train()
        self.embedding_layer.train()
        self.size_classifier.train()
        self.language_classifier.eval()

        self.encoder_optimizer.zero_grad()
        embeddings = self.forward_embeddings(**inputs)
        symb_logits = self.size_classifier(self.dropout(embeddings))
        lang_logits = self.language_classifier(self.dropout(embeddings))

        symb_loss = self.symbolism_loss_fn(symb_logits, symbolism_labels)
        lang_loss = self.language_loss_fn(lang_logits, language_labels)
        total_loss = symb_loss - current_lambda * lang_loss
        total_loss.backward()
        self.encoder_optimizer.step()

        return symb_loss.item(), lang_loss.item()

    def evaluate(self, dataloader, device):
        self.bert.eval()
        self.embedding_layer.eval()
        self.size_classifier.eval()
        self.language_classifier.eval()

        total_symb_correct = 0
        total_lang_correct = 0
        total = 0
        with torch.no_grad():
            for inputs, symb_labels, lang_labels in dataloader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                symb_labels = symb_labels.to(device)
                lang_labels = lang_labels.to(device)

                embeddings = self.forward_embeddings(**inputs)
                symb_preds = self.size_classifier(self.dropout(embeddings)).argmax(dim=1)
                lang_preds = self.language_classifier(self.dropout(embeddings)).argmax(dim=1)

                total_symb_correct += (symb_preds == symb_labels).sum().item()
                total_lang_correct += (lang_preds == lang_labels).sum().item()
                total += symb_labels.size(0)

        return total_symb_correct/total, total_lang_correct/total

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'metadata': {
                'bert_model_name_or_path': self.bert.config.name_or_path,
                'embedding_dim': self.embedding_layer.out_features,
                'num_languages': self.language_classifier.out_features,
                'lambda_param': self.lambda_param
            }
        }, os.path.join(path, 'adv_model.pt'))

    @classmethod
    def load(cls, path):
        ckpt = torch.load(os.path.join(path, 'adv_model.pt'), map_location='cpu')
        meta = ckpt['metadata']
        model = cls(
            bert_model_name_or_path=meta['bert_model_name_or_path'],
            embedding_dim=meta['embedding_dim'],
            num_languages=meta['num_languages'],
            lambda_param=meta['lambda_param']
        )
        model.load_state_dict(ckpt['state_dict'])
        return model

def main():
    class Args:
        bert_model_name_or_path = "bert-ipa-model"
        train_data = "../Data/corpus_clean_train.csv"
        test_data = "../Data/corpus_clean_test.csv"
        output = "debug_model_out"
        epochs = 35
        batch_size = 16
        max_length = 64
        embedding_dim = 16  # was 16 — set to 32 for more representational power
        lambda_param = 0.01  # keep it small to avoid over-scrubbing

    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    print("LABEL CHECK:")
    print(train_df['label'].value_counts())
    print("Label dtype:", train_df['label'].dtype)
    print(train_df[['transcription', 'label']].sample(5))

    num_languages = len(train_df['lang_fam'].unique())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.bert_model_name_or_path)

    train_ds = IPADataset(train_df, tokenizer, args.max_length)
    test_ds = IPADataset(test_df, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = AdversarialSoundSymbolismModel(
        bert_model_name_or_path=args.bert_model_name_or_path,
        embedding_dim=args.embedding_dim,
        num_languages=num_languages,
        lambda_param=args.lambda_param
    ).to(device)

    print("\n=== Beginning sanity training ===")
    for epoch in range(args.epochs):
        total_symb_loss = 0.0
        total_lang_adv_loss = 0.0
        nbatches = 0
        model.train()

        # Lambda annealing: no adv training until epoch 10, then gradually increase
        if epoch < 10:
            current_lambda = 0.0
        else:
            current_lambda = args.lambda_param * (1 - 0.95 ** (epoch - 9))

        for i, (inputs, symb_labels, lang_labels) in enumerate(train_loader):
            nbatches += 1
            inputs = {k: v.to(device) for k, v in inputs.items()}
            symb_labels = symb_labels.to(device)
            lang_labels = lang_labels.to(device)

            if epoch == 0 and i == 0:
                print("\nFirst batch input_ids:", inputs["input_ids"][0])
                print("Corresponding label:", symb_labels[0])

            clf_loss = model.train_language_classifier(inputs, lang_labels)
            symb_loss, lang_adv_loss = model.train_encoder_symbolism(
                inputs, symb_labels, lang_labels, current_lambda
            )

            if i == 0:
                logits = model.size_classifier(model.dropout(model.forward_embeddings(**inputs)))
                print(f"\n[Epoch {epoch+1}] Raw logits from classifier:", logits[0].detach().cpu().numpy())

            total_symb_loss += symb_loss
            total_lang_adv_loss += lang_adv_loss

        symb_acc, lang_acc = model.evaluate(test_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | SymbLoss={total_symb_loss/nbatches:.4f} "
              f"LangAdvLoss={total_lang_adv_loss/nbatches:.4f} "
              f"TestSymbAcc={symb_acc:.4f} TestLangAcc={lang_acc:.4f}")

    print(f"\nNumber of batches per epoch: {nbatches}")
    print("Finished sanity check training.\n")


if __name__ == '__main__':
    main()