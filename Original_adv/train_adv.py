import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
import re
import unicodedata
from collections import Counter
from torch.utils.data import DataLoader, Dataset # Added Dataset for clarity

# --- Model Components (Unchanged) ---
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class SymbolismClassifier(nn.Module):
    def __init__(self, embedding_dim=32):
        super(SymbolismClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Binary classification: large/small
        )

    def forward(self, embeddings):
        return self.network(embeddings)


class LanguageClassifier(nn.Module):
    def __init__(self, embedding_dim=32, num_languages=3):
        super(LanguageClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_languages)
        )

    def forward(self, embeddings):
        return self.network(embeddings)

# --- Adversarial Model Class (Modified for Evaluation) ---
class AdversarialSoundSymbolismModel:
    def __init__(self, input_dim, embedding_dim=16, num_languages=3, lambda_param=0.1):
        self.encoder = Encoder(input_dim, embedding_dim)
        self.symbolism_classifier = SymbolismClassifier(embedding_dim)
        self.language_classifier = LanguageClassifier(embedding_dim, num_languages)

        self.lambda_param = lambda_param
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_languages = num_languages

        # Optimizers
        self.encoder_optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.symbolism_classifier.parameters()),
            lr=0.001,
            weight_decay=1e-5
        )
        self.language_classifier_optimizer = optim.Adam(
            self.language_classifier.parameters(),
            lr=0.001
        )

        # Loss functions
        self.symbolism_loss_fn = nn.CrossEntropyLoss()
        self.language_loss_fn = nn.CrossEntropyLoss()

    def train_language_classifier(self, inputs, language_labels):
        # Freeze encoder
        self.encoder.eval()
        # Symbolism classifier is not involved, but good practice to set mode
        self.symbolism_classifier.eval()
        self.language_classifier.train()

        # Zero gradients
        self.language_classifier_optimizer.zero_grad()

        # Get embeddings (detached to prevent gradient flow to encoder)
        with torch.no_grad():
            embeddings = self.encoder(inputs)

        # Predict languages
        language_predictions = self.language_classifier(embeddings)

        # Compute language classification loss
        language_loss = self.language_loss_fn(language_predictions, language_labels)
        language_loss.backward()

        # Update language classifier
        self.language_classifier_optimizer.step()

        return language_loss.item()

    def train_encoder_symbolism(self, inputs, symbolism_labels, language_labels):
        # Unfreeze encoder and symbolism classifier
        self.encoder.train()
        self.symbolism_classifier.train()
        # Freeze adversary
        self.language_classifier.eval()

        # Zero gradients
        self.encoder_optimizer.zero_grad()

        # Get embeddings
        embeddings = self.encoder(inputs)

        # Predict symbolism
        symbolism_predictions = self.symbolism_classifier(embeddings)

        # Predict languages (gradients flow back to encoder but not adversary weights)
        language_predictions = self.language_classifier(embeddings)

        # Compute symbolism loss
        symbolism_loss = self.symbolism_loss_fn(symbolism_predictions, symbolism_labels)

        # Compute language loss (for adversarial objective)
        # Note: Gradients from this loss WILL flow back to the encoder
        language_loss_adv = self.language_loss_fn(language_predictions, language_labels)

        # Compute total loss (with adversarial term)
        total_loss = symbolism_loss - self.lambda_param * language_loss_adv
        total_loss.backward()

        # Update encoder and symbolism classifier
        self.encoder_optimizer.step()
        return symbolism_loss.item(), language_loss_adv.item()

    # --- NEW: Evaluation Method ---
    def evaluate(self, dataloader):
        """Evaluate the model on a given dataloader."""
        self.encoder.eval()
        self.symbolism_classifier.eval()
        self.language_classifier.eval()

        total_symbolism_correct = 0
        total_language_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, symbolism_labels, language_labels = batch
                total_samples += inputs.size(0)

                # Get embeddings
                embeddings = self.encoder(inputs)

                # Symbolism predictions
                symbolism_logits = self.symbolism_classifier(embeddings)
                symbolism_preds = torch.argmax(symbolism_logits, dim=1)
                total_symbolism_correct += (symbolism_preds == symbolism_labels).sum().item()

                # Language predictions
                language_logits = self.language_classifier(embeddings)
                language_preds = torch.argmax(language_logits, dim=1)
                total_language_correct += (language_preds == language_labels).sum().item()

        symbolism_accuracy = total_symbolism_correct / total_samples if total_samples > 0 else 0
        language_accuracy = total_language_correct / total_samples if total_samples > 0 else 0

        return symbolism_accuracy, language_accuracy

    def train(self, train_dataloader, test_dataloader, epochs=100):
        for epoch in range(epochs):
            # --- Training Phase ---
            self.encoder.train() # Ensure models start in train mode
            self.symbolism_classifier.train()
            self.language_classifier.train() # Adversary needs to train too

            total_symbolism_loss = 0
            total_language_adv_loss = 0 # Adversarial loss seen by encoder
            total_language_clf_loss = 0 # Loss when training the classifier itself
            num_batches = 0

            for batch in train_dataloader:
                inputs, symbolism_labels, language_labels = batch
                num_batches += 1

                # Step A: Train language classifier
                language_loss_clf = self.train_language_classifier(inputs, language_labels)
                total_language_clf_loss += language_loss_clf # Add classifier's loss

                # Step B: Train encoder and symbolism classifier
                symbolism_loss, language_adv_loss = self.train_encoder_symbolism(
                    inputs, symbolism_labels, language_labels
                )

                total_symbolism_loss += symbolism_loss
                total_language_adv_loss += language_adv_loss

            avg_symbolism_loss = total_symbolism_loss / num_batches
            avg_language_adv_loss = total_language_adv_loss / num_batches
            avg_language_clf_loss = total_language_clf_loss / num_batches # Added

            # --- Evaluation Phase ---
            symbolism_acc, language_acc = self.evaluate(test_dataloader)

            # Print epoch statistics
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Symb Loss = {avg_symbolism_loss:.4f}, "
                  f"Lang Adv Loss = {avg_language_adv_loss:.4f}, " # Loss E tries to MAXIMIZE
                  f"Lang Clf Loss = {avg_language_clf_loss:.4f}, " # Loss A tries to MINIMIZE
                  f"Test Symb Acc = {symbolism_acc:.4f}, "
                  f"Test Lang Acc = {language_acc:.4f}")


    def predict_symbolism(self, inputs):
        self.encoder.eval()
        self.symbolism_classifier.eval()

        with torch.no_grad():
            embeddings = self.encoder(inputs)
            symbolism_predictions = self.symbolism_classifier(embeddings)
            return torch.softmax(symbolism_predictions, dim=1)


    def predict_language(self, inputs):
        self.encoder.eval()
        self.language_classifier.eval()

        with torch.no_grad():
            embeddings = self.encoder(inputs)
            language_predictions = self.language_classifier(embeddings)
            return torch.softmax(language_predictions, dim=1)

    def get_language_invariant_embeddings(self, inputs):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(inputs)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        metadata = {
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'num_languages': self.num_languages,
            'lambda_param': self.lambda_param
        }
        torch.save(metadata, os.path.join(path, 'metadata.pt'))
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pt'))
        torch.save(self.symbolism_classifier.state_dict(), os.path.join(path, 'symbolism_classifier.pt'))
        torch.save(self.language_classifier.state_dict(), os.path.join(path, 'language_classifier.pt'))
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        metadata = torch.load(os.path.join(path, 'metadata.pt'))
        model = cls(
            input_dim=metadata['input_dim'],
            embedding_dim=metadata['embedding_dim'],
            num_languages=metadata['num_languages'],
            lambda_param=metadata['lambda_param']
        )
        model.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pt')))
        model.symbolism_classifier.load_state_dict(torch.load(os.path.join(path, 'symbolism_classifier.pt')))
        model.language_classifier.load_state_dict(torch.load(os.path.join(path, 'language_classifier.pt')))
        print(f"Model loaded from {path}")
        return model

# --- Data Preprocessing Functions (Modified to reuse vocabulary) ---

def create_ipa_vocabulary(transcriptions):
    all_symbols = []
    for transcription in transcriptions:
        symbols = list(transcription)
        all_symbols.extend(symbols)
    vocabulary = sorted(set(all_symbols))
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(vocabulary)}
    return symbol_to_idx, vocabulary

def create_bow_representation(transcription, symbol_to_idx):
    symbol_counts = Counter(transcription)
    bow_vector = torch.zeros(len(symbol_to_idx))
    for symbol, count in symbol_counts.items():
        if symbol in symbol_to_idx: # Ignore symbols not in training vocab
            bow_vector[symbol_to_idx[symbol]] = count
    return bow_vector

def load_and_preprocess_data(file_path, symbol_to_idx=None):
    """
    Load and preprocess data. If symbol_to_idx is None, create a new vocabulary.
    Otherwise, use the provided vocabulary.
    """
    df = pd.read_csv(file_path)

    create_new_vocab = symbol_to_idx is None
    if create_new_vocab:
        symbol_to_idx, vocabulary = create_ipa_vocabulary(df['transcription'])
    else:
        vocabulary = None # Not needed when using existing vocab

    X = torch.stack([create_bow_representation(t, symbol_to_idx)
                     for t in df['transcription']])

    symbolism_labels = torch.tensor(df['label'].values, dtype=torch.long) # Ensure LongTensor for CrossEntropy
    # TODO change lang_fam back to lang_label
    language_labels = torch.tensor(df['lang_fam'].values, dtype=torch.long) # Ensure LongTensor

    num_languages = None
    if create_new_vocab:
        # Determine num_languages only when creating vocab from training data
        languages = df['lang_name'].unique()
        num_languages = len(languages)

    if create_new_vocab:
        return X, symbolism_labels, language_labels, num_languages, symbol_to_idx, vocabulary
    else:
        # When using existing vocab, don't return vocab-related info again
        return X, symbolism_labels, language_labels


# --- Simple Dataset Wrapper ---
class IPADataset(Dataset):
    def __init__(self, features, symbolism_labels, language_labels):
        self.features = features
        self.symbolism_labels = symbolism_labels
        self.language_labels = language_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.symbolism_labels[idx], self.language_labels[idx]


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Train Adversarial Sound Symbolism Model')
    parser.add_argument('--data', type=str, required=True, help='Path to the training CSV data file')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test CSV data file')
    parser.add_argument('--output', type=str, default='model', help='Path to save the model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and testing')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of embeddings')
    parser.add_argument('--lambda_param', type=float, default=0.05, help='Weight for adversarial loss')

    args = parser.parse_args()

    # --- Load and preprocess training data ---
    print("Loading training data...")
    X_train, y_symb_train, y_lang_train, num_languages, symbol_to_idx, vocabulary = load_and_preprocess_data(
        args.data, symbol_to_idx=None # Create vocab here
    )
    print(f"Training data loaded. Vocabulary size: {len(symbol_to_idx)}, Num languages: {num_languages}")

    # Save vocabulary for inference and consistency
    os.makedirs(args.output, exist_ok=True)
    torch.save({
        'symbol_to_idx': symbol_to_idx,
        'vocabulary': vocabulary,
        'num_languages': num_languages,
    }, os.path.join(args.output, 'vocabulary.pt'))
    print(f"Vocabulary saved to {os.path.join(args.output, 'vocabulary.pt')}")

    # --- Load and preprocess test data using the training vocabulary ---
    print("Loading test data...")
    X_test, y_symb_test, y_lang_test = load_and_preprocess_data(
        args.test_data, symbol_to_idx=symbol_to_idx # Reuse training vocab
    )
    print("Test data loaded.")

    # --- Create datasets and dataloaders ---
    train_dataset = IPADataset(X_train, y_symb_train, y_lang_train)
    test_dataset = IPADataset(X_test, y_symb_test, y_lang_test)

    # Shuffle training data, no need to shuffle test data
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # --- Initialize and train model ---
    print("Initializing model...")
    model = AdversarialSoundSymbolismModel(
        input_dim=len(symbol_to_idx),
        embedding_dim=args.embedding_dim,
        num_languages=num_languages,
        lambda_param=args.lambda_param
    )
    print("Starting training...")
    # Pass test_dataloader to the train method
    model.train(train_dataloader, test_dataloader, epochs=args.epochs)

    # --- Save the final model ---
    print("Training finished. Saving model...")
    model.save(args.output)

if __name__ == "__main__":
    main()