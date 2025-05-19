import argparse, os, re, unicodedata
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader, Dataset

# --- Model Components (Reused/Adapted from original script) ---
class Encoder(nn.Module):
    """Encodes IPA features into embeddings."""
    def __init__(self, input_dim, embedding_dim=32):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

class SymbolismClassifier(nn.Module):
    """Predicts sound symbolism (large/small) from embeddings."""
    def __init__(self, embedding_dim=32):
        super(SymbolismClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Binary classification: large/small (2 classes)
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
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(vocabulary)}
    return symbol_to_idx, vocabulary

def create_bow_representation(transcription, symbol_to_idx):
    """Create bag-of-words representation for an IPA transcription"""
    symbol_counts = Counter(transcription)
    bow_vector = torch.zeros(len(symbol_to_idx))
    for symbol, count in symbolimport argparse, os, re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader, Dataset

# --- Model Component (Simple Logistic Regression) ---
class LogisticRegressionClassifier(nn.Module):
    """Simple logistic regression model that works directly on bag-of-words features."""
    def __init__(self, input_dim):
        super(LogisticRegressionClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 2)  # Binary classification: large/small (2 classes)
    
    def forward(self, x):
        return self.linear(x)

# --- Data Preprocessing Functions (Reused) ---
def create_ipa_vocabulary(transcriptions):
    """Create a vocabulary of IPA symbols from transcriptions"""
    all_symbols = []
    for transcription in transcriptions:
        symbols = list(transcription)
        all_symbols.extend(symbols)
    vocabulary = sorted(set(all_symbols))
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
    Otherwise, use the provided vocabulary. Returns only features and symbolism labels.
    """
    df = pd.read_csv(file_path)

    create_new_vocab = symbol_to_idx is None
    if create_new_vocab:
        symbol_to_idx, vocabulary = create_ipa_vocabulary(df['transcription'])
    else:
        vocabulary = None # Not needed when using existing vocab

    X = torch.stack([create_bow_representation(t, symbol_to_idx)
                     for t in df['transcription']])

    # Only need symbolism labels for this script
    symbolism_labels = torch.tensor(df['label'].values, dtype=torch.long)

    if create_new_vocab:
        return X, symbolism_labels, symbol_to_idx, vocabulary
    else:
        return X, symbolism_labels

# --- Simple Dataset Wrapper ---
class IPASymbolismDataset(Dataset):
    def __init__(self, features, symbolism_labels):
        self.features = features
        self.symbolism_labels = symbolism_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.symbolism_labels[idx]

# --- Evaluation Function ---
def evaluate(model, dataloader, loss_fn):
    """Evaluate the logistic regression classification performance."""
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, symbolism_labels = batch
            total_samples += inputs.size(0)

            # Forward pass
            logits = model(inputs)

            # Calculate loss
            loss = loss_fn(logits, symbolism_labels)
            total_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == symbolism_labels).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return avg_loss, accuracy

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Train Logistic Regression Sound Symbolism Classifier on IPA Features')
    parser.add_argument('--data', type=str, required=True, help='Path to the training CSV data file')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test CSV data file')
    parser.add_argument('--output', type=str, default='symbolism_model', help='Path to save the model components')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and testing')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')

    args = parser.parse_args()

    # --- Load and preprocess training data ---
    print("Loading training data...")
    X_train, y_symb_train, symbol_to_idx, vocabulary = load_and_preprocess_data(
        args.data, symbol_to_idx=None # Create vocab here
    )
    print(f"Training data loaded. Vocabulary size: {len(symbol_to_idx)}")

    # Save vocabulary for consistency and potential later use
    os.makedirs(args.output, exist_ok=True)
    torch.save({
        'symbol_to_idx': symbol_to_idx,
        'vocabulary': vocabulary,
    }, os.path.join(args.output, 'vocabulary.pt'))
    print(f"Vocabulary saved to {os.path.join(args.output, 'vocabulary.pt')}")

    # --- Load and preprocess test data using the training vocabulary ---
    print("Loading test data...")
    X_test, y_symb_test = load_and_preprocess_data(
        args.test_data, symbol_to_idx=symbol_to_idx # Reuse training vocab
    )
    print("Test data loaded.")

    # --- Create datasets and dataloaders ---
    train_dataset = IPASymbolismDataset(X_train, y_symb_train)
    test_dataset = IPASymbolismDataset(X_test, y_symb_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Initialize model ---
    print("Initializing logistic regression model...")
    model = LogisticRegressionClassifier(input_dim=len(symbol_to_idx))

    # --- Define optimizer and loss function ---
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0

        for batch in train_dataloader:
            inputs, symbolism_labels = batch
            num_batches += 1

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)

            # Calculate loss
            loss = loss_fn(logits, symbolism_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0

        # --- Evaluation Phase ---
        avg_test_loss, test_accuracy = evaluate(model, test_dataloader, loss_fn)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Test Loss = {avg_test_loss:.4f}, "
              f"Test Accuracy = {test_accuracy:.4f}")

    # --- Save the final model ---
    print("Training finished. Saving model...")
    os.makedirs(args.output, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output, 'logistic_regression_model.pt'))
    
    # Save metadata
    metadata = {
        'input_dim': len(symbol_to_idx),
        'model_type': 'logistic_regression'
    }
    torch.save(metadata, os.path.join(args.output, 'metadata.pt'))
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()_counts.items():
        if symbol in symbol_to_idx: # Ignore symbols not in training vocab
            bow_vector[symbol_to_idx[symbol]] = count
    return bow_vector

def load_and_preprocess_data(file_path, symbol_to_idx=None):
    """
    Load and preprocess data. If symbol_to_idx is None, create a new vocabulary.
    Otherwise, use the provided vocabulary. Returns only features and symbolism labels.
    """
    df = pd.read_csv(file_path)

    create_new_vocab = symbol_to_idx is None
    if create_new_vocab:
        symbol_to_idx, vocabulary = create_ipa_vocabulary(df['transcription'])
    else:
        vocabulary = None # Not needed when using existing vocab

    X = torch.stack([create_bow_representation(t, symbol_to_idx)
                     for t in df['transcription']])

    # Only need symbolism labels for this script
    symbolism_labels = torch.tensor(df['label'].values, dtype=torch.long) # Ensure LongTensor

    if create_new_vocab:
        return X, symbolism_labels, symbol_to_idx, vocabulary
    else:
        return X, symbolism_labels

# --- Simple Dataset Wrapper (Only Symbolism Labels) ---
class IPASymbolismDataset(Dataset):
    def __init__(self, features, symbolism_labels):
        self.features = features
        self.symbolism_labels = symbolism_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.symbolism_labels[idx]

# --- Evaluation Function ---
def evaluate(encoder, symbolism_classifier, dataloader, loss_fn):
    """Evaluate the sound symbolism classification performance."""
    encoder.eval()
    symbolism_classifier.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, symbolism_labels = batch
            total_samples += inputs.size(0)

            # Forward pass
            embeddings = encoder(inputs)
            symbolism_logits = symbolism_classifier(embeddings)

            # Calculate loss
            loss = loss_fn(symbolism_logits, symbolism_labels)
            total_loss += loss.item() * inputs.size(0) # Accumulate total loss

            # Calculate accuracy
            symbolism_preds = torch.argmax(symbolism_logits, dim=1)
            total_correct += (symbolism_preds == symbolism_labels).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return avg_loss, accuracy

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Train Simple Sound Symbolism Classifier on IPA Features')
    parser.add_argument('--data', type=str, required=True, help='Path to the training CSV data file')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test CSV data file')
    parser.add_argument('--output', type=str, default='symbolism_model', help='Path to save the model components')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and testing')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of embeddings')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')

    args = parser.parse_args()

    # --- Load and preprocess training data ---
    print("Loading training data...")
    X_train, y_symb_train, symbol_to_idx, vocabulary = load_and_preprocess_data(
        args.data, symbol_to_idx=None # Create vocab here
    )
    print(f"Training data loaded. Vocabulary size: {len(symbol_to_idx)}")

    # Save vocabulary for consistency and potential later use
    os.makedirs(args.output, exist_ok=True)
    # Note: num_languages is not relevant, so not saved in vocab file here
    torch.save({
        'symbol_to_idx': symbol_to_idx,
        'vocabulary': vocabulary,
    }, os.path.join(args.output, 'vocabulary.pt'))
    print(f"Vocabulary saved to {os.path.join(args.output, 'vocabulary.pt')}")

    # --- Load and preprocess test data using the training vocabulary ---
    print("Loading test data...")
    X_test, y_symb_test = load_and_preprocess_data(
        args.test_data, symbol_to_idx=symbol_to_idx # Reuse training vocab
    )
    print("Test data loaded.")

    # --- Create datasets and dataloaders ---
    train_dataset = IPASymbolismDataset(X_train, y_symb_train)
    test_dataset = IPASymbolismDataset(X_test, y_symb_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Initialize models ---
    print("Initializing models...")
    encoder = Encoder(input_dim=len(symbol_to_idx), embedding_dim=args.embedding_dim)
    symbolism_classifier = SymbolismClassifier(embedding_dim=args.embedding_dim) # Note: No num_classes needed

    # --- Define optimizer and loss function ---
    # Optimizer updates BOTH encoder and classifier
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(symbolism_classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        encoder.train()
        symbolism_classifier.train()
        total_train_loss = 0
        num_batches = 0

        for batch in train_dataloader:
            inputs, symbolism_labels = batch
            num_batches += 1

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            embeddings = encoder(inputs)
            symbolism_logits = symbolism_classifier(embeddings)

            # Calculate loss
            loss = loss_fn(symbolism_logits, symbolism_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0

        # --- Evaluation Phase ---
        avg_test_loss, test_accuracy = evaluate(encoder, symbolism_classifier, test_dataloader, loss_fn)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Test Loss = {avg_test_loss:.4f}, "
              f"Test Accuracy = {test_accuracy:.4f}")

    # --- Save the final models ---
    print("Training finished. Saving model components...")
    os.makedirs(args.output, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(args.output, 'encoder.pt'))
    torch.save(symbolism_classifier.state_dict(), os.path.join(args.output, 'symbolism_classifier.pt'))
    # Save metadata matching the vocabulary file for completeness
    metadata = {
        'input_dim': len(symbol_to_idx),
        'embedding_dim': args.embedding_dim,
        # num_classes for symbolism is fixed at 2, could add if desired
    }
    torch.save(metadata, os.path.join(args.output, 'metadata.pt'))
    print(f"Model components saved to {args.output}")


if __name__ == "__main__":
    main()