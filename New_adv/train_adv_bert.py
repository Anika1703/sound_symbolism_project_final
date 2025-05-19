import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import BertModel, BertTokenizerFast


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
    """
    Dataset wrapping transcriptions and labels, tokenized with BERT tokenizer.
    Returns: inputs_dict, symbolism_label, language_label
    """
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
    def __init__(
        self,
        bert_model_name_or_path: str,
        embedding_dim: int,
        num_languages: int,
        lambda_param: float
    ):
        super().__init__()
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(bert_model_name_or_path)
        hidden_size = self.bert.config.hidden_size
        # Single linear layers for projection and classification
        self.embedding_layer = nn.Linear(hidden_size, embedding_dim)
        self.size_classifier = nn.Linear(embedding_dim, 2)
        self.language_classifier = nn.Linear(embedding_dim, num_languages)

        self.lambda_param = lambda_param

        # Optimizers: one for encoder+projection+size head, one for language head
        self.encoder_optimizer = optim.Adam(
            list(self.bert.parameters()) +
            list(self.embedding_layer.parameters()) +
            list(self.size_classifier.parameters()),
            lr=2e-5, weight_decay=1e-5
        )
        self.language_optimizer = optim.Adam(
            self.language_classifier.parameters(), lr=1e-4
        )

        # Loss functions
        self.symbolism_loss_fn = nn.CrossEntropyLoss()
        self.language_loss_fn = nn.CrossEntropyLoss()

    def forward_embeddings(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token pooling
        pooled = outputs.pooler_output  # shape: (batch, hidden)
        embeddings = self.embedding_layer(pooled)
        return embeddings

    def train_language_classifier(self, inputs, language_labels):
        # Freeze BERT, embedding, and size head
        self.bert.eval()
        self.embedding_layer.eval()
        self.size_classifier.eval()
        self.language_classifier.train()

        self.language_optimizer.zero_grad()
        # Detached embeddings
        embeddings = self.forward_embeddings(**inputs).detach()
        # Predict language
        lang_logits = self.language_classifier(embeddings)
        loss = self.language_loss_fn(lang_logits, language_labels)
        loss.backward()
        self.language_optimizer.step()
        return loss.item()

    def train_encoder_symbolism(self, inputs, symbolism_labels, language_labels):
        # Unfreeze BERT, embedding, and size head; freeze language head
        self.bert.train()
        self.embedding_layer.train()
        self.size_classifier.train()
        self.language_classifier.eval()

        self.encoder_optimizer.zero_grad()
        embeddings = self.forward_embeddings(**inputs)
        # Symbolism prediction
        symb_logits = self.size_classifier(embeddings)
        # Language prediction
        lang_logits = self.language_classifier(embeddings)

        symb_loss = self.symbolism_loss_fn(symb_logits, symbolism_labels)
        lang_loss = self.language_loss_fn(lang_logits, language_labels)
        # Adversarial objective: maximize language loss
        total_loss = symb_loss - self.lambda_param * lang_loss
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
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                symb_labels = symb_labels.to(device)
                lang_labels = lang_labels.to(device)

                embeddings = self.forward_embeddings(**inputs)
                symb_preds = self.size_classifier(embeddings).argmax(dim=1)
                lang_preds = self.language_classifier(embeddings).argmax(dim=1)

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
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    num_languages = len(train_df['lang_fam'].unique())

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name_or_path)

    # Datasets & Dataloaders
    train_ds = IPADataset(train_df, tokenizer, args.max_length)
    test_ds  = IPADataset(test_df,  tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # Model
    model = AdversarialSoundSymbolismModel(
        bert_model_name_or_path=args.bert_model_name_or_path,
        embedding_dim=args.embedding_dim,
        num_languages=num_languages,
        lambda_param=args.lambda_param
    ).to(device)

    # Training loop
    for epoch in range(args.epochs):
        total_symb_loss = 0.0
        total_lang_adv_loss = 0.0
        total_lang_clf_loss = 0.0
        nbatches = 0
        model.train()
        for inputs, symb_labels, lang_labels in train_loader:
            nbatches += 1
            # Move labels
            symb_labels = symb_labels.to(device)
            lang_labels = lang_labels.to(device)
            # Train adversary
            clf_loss = model.train_language_classifier(
                {k: v.to(device) for k, v in inputs.items()}, lang_labels
            )
            # Train embedding + symbolism
            symb_loss, lang_adv_loss = model.train_encoder_symbolism(
                {k: v.to(device) for k, v in inputs.items()},
                symb_labels,
                lang_labels
            )
            total_lang_clf_loss += clf_loss
            total_symb_loss += symb_loss
            total_lang_adv_loss += lang_adv_loss

        # Evaluation
        symb_acc, lang_acc = model.evaluate(test_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | SymbLoss={total_symb_loss/nbatches:.4f} "
              f"LangClfLoss={total_lang_clf_loss/nbatches:.4f} "
              f"LangAdvLoss={total_lang_adv_loss/nbatches:.4f} "
              f"TestSymbAcc={symb_acc:.4f} TestLangAcc={lang_acc:.4f}")

    # Save model
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == '__main__':
    main()