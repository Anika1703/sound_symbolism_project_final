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
from torch.utils.data import DataLoader, Dataset

# --- Model Components (Reused from original script) ---
class Encoder(nn.Module):
    """Encodes IPA features into embeddings."""
    def __init__(self, input_dim, embedding_dim=16):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class LanguageClassifier(nn.Module):
    """Predicts language from embeddings."""
    def __init__(self, embedding_dim=16, num_languages=3):
        super(LanguageClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2), # Can include dropout if desired
            nn.Linear(16, num_languages)
        )

    def forward(self, embeddings):
        return self.network(embeddings)

# --- Data Preprocessing Functions (Reused) ---

def create_ipa_vocabulary(transcriptions):
    """Create a vocabulary of IPA symbols from transcriptions"""
    all_symbols = []
    for transcription in transcriptions:
        symbols = list(transcription)
        all_symbols.extend(symbols)
    vocabulary = sorted(set(all_symbols))
    print(f"Calculated size of vocabulary: {len(vocabulary)}")
    print(f"{vocabulary=}")
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(vocabulary)}
    return symbol_to_idx, vocabulary


def create_bow_representation(transcription, symbol_to_idx):
    """Create bag-of-words representation for an IPA transcription"""
    symbol_counts = Counter(transcription)
    bow_vector = torch.zeros(len(symbol_to_idx))
    for symbol, count in symbol_counts.items():
        if symbol in symbol_to_idx: # Ignore symbols not in training vocab
            bow_vector[symbol_to_idx[symbol]] = count
    return bow_vector


def load_and_preprocess_data(file_path, symbol_to_idx=None):
    """
    Load and preprocess data. If symbol_to_idx is None, create a new vocabulary.
    Otherwise, use the provided vocabulary. Returns only features and language labels.
    """
    df = pd.read_csv(file_path)

    create_new_vocab = symbol_to_idx is None
    if create_new_vocab:
        symbol_to_idx, vocabulary = create_ipa_vocabulary(df['transcription'])
    else:
        vocabulary = None # Not needed when using existing vocab

    X = torch.stack([create_bow_representation(t, symbol_to_idx)
                     for t in df['transcription']])

    # Only need language labels for this script
    # TODO change back to lang label
    language_labels = torch.tensor(df['lang_fam'].values, dtype=torch.long) # Ensure LongTensor

    num_languages = None
    if create_new_vocab:
        # Determine num_languages only when creating vocab from training data
        # Use max label + 1 in case some languages are only in test set (though unlikely)
        num_languages = language_labels.max().item() + 1
        print(f"Num languages calculated to be: {num_languages}")

    if create_new_vocab:
        return X, language_labels, num_languages, symbol_to_idx, vocabulary
    else:
        return X, language_labels


# --- Simple Dataset Wrapper (Only Language Labels) ---
class IPALanguageDataset(Dataset):
    def __init__(self, features, language_labels):
        self.features = features
        self.language_labels = language_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.language_labels[idx]


# --- Evaluation Function ---
def evaluate(encoder, language_classifier, dataloader, loss_fn):
    """Evaluate the language classification performance."""
    encoder.eval()
    language_classifier.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, language_labels = batch
            total_samples += inputs.size(0)

            # Forward pass
            embeddings = encoder(inputs)
            language_logits = language_classifier(embeddings)

            # Calculate loss
            loss = loss_fn(language_logits, language_labels)
            total_loss += loss.item() * inputs.size(0) # Accumulate total loss

            # Calculate accuracy
            language_preds = torch.argmax(language_logits, dim=1)
            total_correct += (language_preds == language_labels).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return avg_loss, accuracy

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Train Simple Language Classifier on IPA Features')
    parser.add_argument('--data', type=str, required=True, help='Path to the training CSV data file')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test CSV data file')
    parser.add_argument('--output', type=str, default='language_model', help='Path to save the model components')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and testing')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Dimension of embeddings')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')

    args = parser.parse_args()

    # --- Load and preprocess training data ---
    print("Loading training data...")
    X_train, y_lang_train, num_languages, symbol_to_idx, vocabulary = load_and_preprocess_data(
        args.data, symbol_to_idx=None # Create vocab here
    )
    print(f"Training data loaded. Vocabulary size: {len(symbol_to_idx)}, Num languages: {num_languages}")

    # Save vocabulary for consistency and potential later use
    os.makedirs(args.output, exist_ok=True)
    torch.save({
        'symbol_to_idx': symbol_to_idx,
        'vocabulary': vocabulary,
        'num_languages': num_languages,
    }, os.path.join(args.output, 'vocabulary.pt'))
    print(f"Vocabulary saved to {os.path.join(args.output, 'vocabulary.pt')}")

    # --- Load and preprocess test data using the training vocabulary ---
    print("Loading test data...")
    X_test, y_lang_test = load_and_preprocess_data(
        args.test_data, symbol_to_idx=symbol_to_idx # Reuse training vocab
    )
    print("Test data loaded.")

    # --- Create datasets and dataloaders ---
    train_dataset = IPALanguageDataset(X_train, y_lang_train)
    test_dataset = IPALanguageDataset(X_test, y_lang_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Initialize models ---
    print("Initializing models...")
    encoder = Encoder(input_dim=len(symbol_to_idx), embedding_dim=args.embedding_dim)
    language_classifier = LanguageClassifier(embedding_dim=args.embedding_dim, num_languages=num_languages)

    # --- Define optimizer and loss function ---
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(language_classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        encoder.train()
        language_classifier.train()
        total_train_loss = 0
        num_batches = 0

        for batch in train_dataloader:
            inputs, language_labels = batch
            num_batches += 1

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            embeddings = encoder(inputs)
            language_logits = language_classifier(embeddings)

            # Calculate loss
            loss = loss_fn(language_logits, language_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0

        # --- Evaluation Phase ---
        avg_test_loss, test_accuracy = evaluate(encoder, language_classifier, test_dataloader, loss_fn)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Test Loss = {avg_test_loss:.4f}, "
              f"Test Accuracy = {test_accuracy:.4f}")

    # --- Save the final models ---
    print("Training finished. Saving model components...")
    os.makedirs(args.output, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(args.output, 'encoder.pt'))
    torch.save(language_classifier.state_dict(), os.path.join(args.output, 'language_classifier.pt'))
    # Save metadata matching the vocabulary file for completeness
    metadata = {
        'input_dim': len(symbol_to_idx),
        'embedding_dim': args.embedding_dim,
        'num_languages': num_languages,
    }
    torch.save(metadata, os.path.join(args.output, 'metadata.pt'))
    print(f"Model components saved to {args.output}")


if __name__ == "__main__":
    main()