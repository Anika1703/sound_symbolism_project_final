import torch
import pandas as pd
import argparse
import os
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import unicodedata
import numpy as np

# Import the model and preprocessing functions from the training script
from train_adv import (
    AdversarialSoundSymbolismModel, 
    create_bow_representation, 
)

def evaluate_model(model, X, symbolism_labels, language_labels):
    """Evaluate model performance on both tasks"""
    # Predict symbolism
    symbolism_probs = model.predict_symbolism(X)
    symbolism_preds = torch.argmax(symbolism_probs, dim=1).cpu().numpy()
    
    # Predict language
    language_probs = model.predict_language(X)
    language_preds = torch.argmax(language_probs, dim=1).cpu().numpy()
    
    # Calculate accuracies
    symbolism_acc = accuracy_score(symbolism_labels.cpu().numpy(), symbolism_preds)
    language_acc = accuracy_score(language_labels.cpu().numpy(), language_preds)
    
    # Detailed reports
    symbolism_report = classification_report(symbolism_labels.cpu().numpy(), symbolism_preds)
    language_report = classification_report(language_labels.cpu().numpy(), language_preds)
    
    return {
        'symbolism_accuracy': symbolism_acc,
        'language_accuracy': language_acc,
        'symbolism_predictions': symbolism_preds,
        'language_predictions': language_preds,
        'symbolism_report': symbolism_report,
        'language_report': language_report
    }

def load_and_preprocess_test_data(file_path, vocabulary_path):
    """Load and preprocess test data using vocabulary from training"""
    # Load vocabulary
    vocab_data = torch.load(vocabulary_path)
    symbol_to_idx = vocab_data['symbol_to_idx']
    
    # Load test data
    df = pd.read_csv(file_path)
    
    # Create BoW representations
    X = torch.stack([create_bow_representation(t, symbol_to_idx) 
                     for t in df['transcription']])
    
    # Extract labels
    symbolism_labels = torch.tensor(df['label'].values)
    # TODO change lang_fam back to lang_label
    language_labels = torch.tensor(df['lang_fam'].values)
    
    return X, symbolism_labels, language_labels, df['transcription'].values

def main():
    parser = argparse.ArgumentParser(description='Evaluate Adversarial Sound Symbolism Model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to the test CSV data file')
    parser.add_argument('--output', type=str, default=None, help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Load vocabulary
    vocabulary_path = os.path.join(args.model, 'vocabulary.pt')
    
    # Load and preprocess test data
    X, symbolism_labels, language_labels, transcriptions = load_and_preprocess_test_data(
        args.data, vocabulary_path
    )
    
    # Load model
    model = AdversarialSoundSymbolismModel.load(args.model)
    
    # Evaluate model
    results = evaluate_model(model, X, symbolism_labels, language_labels)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nSound Symbolism Classification:")
    print(f"Accuracy: {results['symbolism_accuracy']:.4f}")
    print("\nDetailed Report:")
    print(results['symbolism_report'])
    
    print(f"\nLanguage Classification:")
    print(f"Accuracy: {results['language_accuracy']:.4f}")
    print("\nDetailed Report:")
    print(results['language_report'])
    
    # Get language-invariant embeddings
    embeddings = model.get_language_invariant_embeddings(X).cpu().numpy()
    
    # Save results if output path is specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        
        # Save predictions
        output_df = pd.DataFrame({
            'transcription': transcriptions,
            'true_symbolism_label': symbolism_labels.cpu().numpy(),
            'pred_symbolism_label': results['symbolism_predictions'],
            'true_language_label': language_labels.cpu().numpy(),
            'pred_language_label': results['language_predictions'],
        })
        
        # Save embeddings
        for i in range(embeddings.shape[1]):
            output_df[f'embedding_{i}'] = embeddings[:, i]
        
        output_df.to_csv(os.path.join(args.output, 'predictions.csv'), index=False)
        print(f"\nResults saved to {os.path.join(args.output, 'predictions.csv')}")
    
    # Get examples of words that are correctly/incorrectly classified
    correct_idx = results['symbolism_predictions'] == symbolism_labels.cpu().numpy()
    incorrect_idx = ~correct_idx
    
    print("\nExamples of correctly classified words:")
    for idx in np.where(correct_idx)[0][:5]:  # First 5 examples
        print(f"  {transcriptions[idx]}: True={symbolism_labels[idx].item()}, Pred={results['symbolism_predictions'][idx]}")
    
    print("\nExamples of incorrectly classified words:")
    for idx in np.where(incorrect_idx)[0][:5]:  # First 5 examples
        print(f"  {transcriptions[idx]}: True={symbolism_labels[idx].item()}, Pred={results['symbolism_predictions'][idx]}")

if __name__ == "__main__":
    main()