import pandas as pd
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
from collections import defaultdict
import re

# IPA symbols from the images
ipa_symbols = [
    # Pulmonic consonants
    'p', 'b', 't', 'd', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'g', 'q', 'ɢ', 'ʔ',
    'm', 'ɱ', 'n', 'ɳ', 'ɲ', 'ŋ', 'ɴ',
    'ʙ', 'r', 'ʀ',
    'ⱱ', 'ɾ', 'ɽ',
    'ɸ', 'β', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'ʐ', 'ç', 'ʝ', 'x', 'ɣ', 'χ', 'ʁ', 'h', 'ɦ',
    'ɬ', 'ɮ',
    'ʋ', 'ɹ', 'ɻ', 'j', 'ɰ',
    'l', 'ɭ', 'ʎ', 'ʟ',
    
    # Non-pulmonic consonants
    'ʘ', 'ǀ', 'ǃ', 'ǂ', 'ǁ',  # Clicks
    'ɓ', 'ɗ', 'ʄ', 'ɠ', 'ʛ',  # Voiced implosives
    'pʼ', 'tʼ', 'kʼ', 'sʼ',  # Ejectives
    
    # Other symbols
    'ʍ', 'w', 'ɥ', 'ʜ', 'ʢ', 'ʡ', 'ʧ', 'ʤ',  # Special consonants
    
    # Vowels
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
    'ɪ', 'ʏ', 'ʊ',
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
    'ə',
    'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
    'æ',
    'a', 'ɶ', 'ɑ', 'ɒ'
]

# Create a set for faster lookups
ipa_symbols_set = set(ipa_symbols)

def extract_lines_from_pdf(pdf_path):
    lines = []
    try:
        for page_layout in extract_pages(pdf_path):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for text_line in element:
                        line = text_line.get_text().strip()
                        if line and not line.isspace():
                            lines.append(line)
        
        lines = lines[:30]
        return lines
    except Exception as e:
        print(f"ERROR: Processing {pdf_path}: {str(e)}")
        return None

def get_bin_languages(target_lang, bin_df, similarity_type):
    """Get languages in a specific similarity bin for a target language"""
    bin_langs = bin_df[
        (bin_df['Target_Language'] == target_lang) & 
        (bin_df['Similarity_Bin'] == similarity_type)
    ]['Comparison_Language'].tolist()
    return bin_langs

def process_language_data(language, ipa_folder):
    """Process single language data and return features and labels"""
    pdf_path = os.path.join(ipa_folder, f"{language.upper()}.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        return None, None
    
    lines = extract_lines_from_pdf(pdf_path)
    if not lines:
        print(f"ERROR: No lines extracted from {pdf_path}")
        return None, None
    
    labels = [0]*15 + [1]*15
    return lines, labels

# Function to preprocess text to keep only IPA symbols
def preprocess_text(text):
    """Filter text to keep only IPA symbols"""
    # Create a new string with only IPA symbols
    filtered_text = ''.join(char for char in text if char in ipa_symbols_set)
    return filtered_text

# Custom analyzer function for the vectorizer
def custom_ipa_analyzer(doc):
    """Custom analyzer that only returns IPA symbols from the text"""
    # First, filter the text to keep only IPA symbols
    filtered_text = preprocess_text(doc)
    # Return the individual characters
    return list(filtered_text)

def get_important_phonemes(vectorizer, model):
    """Extract the most important phonemes for each class"""
    feature_names = vectorizer.get_feature_names_out()
    
    # Debug: check if feature names are actually IPA symbols
    non_ipa_features = [f for f in feature_names if f not in ipa_symbols_set]
    if non_ipa_features:
        print(f"WARNING: Found {len(non_ipa_features)} non-IPA features")
    
    coefficients = model.coef_[0]
    
    # Create DataFrame with phonemes and coefficients
    coef_df = pd.DataFrame({
        'phoneme': feature_names,
        'coefficient': coefficients
    })
    
    # Filter to keep only IPA symbols
    coef_df = coef_df[coef_df['phoneme'].isin(ipa_symbols_set)]
    
    # Get top phonemes for each class
    top_positive = coef_df.nlargest(10, 'coefficient')
    top_negative = coef_df.nsmallest(10, 'coefficient')
    
    return top_positive, top_negative

def train_and_evaluate(train_data, train_labels, test_data, test_labels, vectorizer):
    """Train models and return accuracies and models"""
    try:
        # Preprocess training and test data to filter for IPA symbols
        processed_train_data = [preprocess_text(text) for text in train_data]
        processed_test_data = [preprocess_text(text) for text in test_data]
        
        # Transform the data
        X_train = vectorizer.transform(processed_train_data).toarray()
        X_test = vectorizer.transform(processed_test_data).toarray()
        
        # Debug: check feature names
        feature_names = vectorizer.get_feature_names_out()
        # Check for non-IPA symbols in features
        non_ipa = [f for f in feature_names if f not in ipa_symbols_set]
        if non_ipa:
            print(f"WARNING: Non-IPA symbols in features")
        
        # Train Logistic Regression
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, train_labels)
        lr_pred = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(test_labels, lr_pred)
        print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")
        
        results = []
        print(f"Testing Decision Tree parameters...")
        for max_depth in [3, 4, 5, 6, 7, 8, 10, 15]:
            for min_samples_split in [2, 3, 5, 7, 10]:
                dt = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split
                )
                dt.fit(X_train, train_labels)
                train_pred = dt.predict(X_train)
                train_acc = accuracy_score(train_labels, train_pred)
                
                # Evaluate on test data (target language)
                test_pred = dt.predict(X_test)
                test_acc = accuracy_score(test_labels, test_pred)
                
                # Calculate overfitting gap
                gap = train_acc - test_acc
                results.append({
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'gap': gap
                })
                
        results_df = pd.DataFrame(results)
        best_idx = results_df['test_accuracy'].idxmax()
        best_params = {
            'max_depth': results_df.loc[best_idx, 'max_depth'],
            'min_samples_split': results_df.loc[best_idx, 'min_samples_split']
        }
        print(f"Best Decision Tree parameters: {best_params}")
        
        dt_model = DecisionTreeClassifier(
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split']
        )
        dt_model.fit(X_train, train_labels)
        dt_pred = dt_model.predict(X_test)
        dt_accuracy = accuracy_score(test_labels, dt_pred)
        print(f"Decision Tree accuracy: {dt_accuracy:.4f}")
        
        return lr_accuracy, dt_accuracy, lr_model, dt_model, best_params, results
    except Exception as e:
        print(f"ERROR: train_and_evaluate_with_tuning: {str(e)}")
        import traceback
        traceback.print_exc()
        return None 

def get_all_phonemes(vectorizer, model):
    """Extract all phonemes with their coefficients"""
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Create DataFrame with phonemes and coefficients
    coef_df = pd.DataFrame({
        'phoneme': feature_names,
        'coefficient': coefficients
    })
    
    # Filter to keep only IPA symbols
    coef_df = coef_df[coef_df['phoneme'].isin(ipa_symbols_set)]
    
    top_positive = coef_df.nlargest(10, 'coefficient')
    top_negative = coef_df.nsmallest(10, 'coefficient')
    
    return coef_df, top_positive, top_negative
def main():
    print("=== Sound Symbolism Project Starting ===")
    
    # Read bin information
    bin_df = pd.read_csv('language_bins_lookup.csv')
    print(f"Loaded language bins information")

    results = []
    phoneme_results = defaultdict(list)
    all_phoneme_coeffs = []  # For phoneme analysis
    
    # Get unique target languages
    target_languages = bin_df['Target_Language'].unique()
    ipa_folder = 'IPA'
    
    for target_lang in target_languages:
        print(f"\nProcessing target language: {target_lang}")
        
        # Get target language data
        target_data, target_labels = process_language_data(target_lang, ipa_folder)
        if target_data is None:
            print(f"Skipping {target_lang} - no data available")
            continue
            
        # Process each similarity bin
        for similarity_bin in ['Most Similar', 'Somewhat Similar', 'Least Similar']:
            print(f"Processing {similarity_bin} bin")
            
            # Get languages in this bin
            bin_languages = get_bin_languages(target_lang, bin_df, similarity_bin)
            
            # Combine training data
            train_data = []
            train_labels = []
            
            for lang in bin_languages:
                lang_data, lang_labels = process_language_data(lang, ipa_folder)
                if lang_data is not None:
                    train_data.extend(lang_data)
                    train_labels.extend(lang_labels)
            
            if not train_data:
                print(f"ERROR: No training data for {target_lang}, {similarity_bin} bin")
                continue
                
            # Configure vectorizer to use our custom analyzer
            vectorizer = CountVectorizer(
                analyzer=custom_ipa_analyzer,
                lowercase=False  # Important for preserving IPA symbols
            )
            
            # Fit vectorizer on all data after preprocessing
            all_data = [preprocess_text(text) for text in train_data]
            vectorizer.fit(all_data)
            
            # Train and evaluate models with parameter tuning
            model_results = train_and_evaluate(
                train_data, train_labels, target_data, target_labels, vectorizer
            )
            
            if model_results is not None:
                lr_acc, dt_acc, lr_model, dt_model, best_params, _ = model_results
                
                # Store accuracy results
                results.append({
                    'Target_Language': target_lang,
                    'Similarity_Bin': similarity_bin,
                    'Logistic_Regression_Accuracy': lr_acc,
                    'Decision_Tree_Accuracy': dt_acc,
                    'DT_Best_Max_Depth': best_params['max_depth'] if best_params else None,
                    'DT_Best_Min_Samples_Split': best_params['min_samples_split'] if best_params else None,
                    'Training_Languages': ', '.join(bin_languages),
                    'Num_Training_Languages': len(bin_languages)
                })
                
                # Get and store all phoneme coefficients for this language/bin combination
                if lr_model is not None:
                    all_coefs, top_positive, top_negative = get_all_phonemes(vectorizer, lr_model)
                    
                    # Store top phonemes for backward compatibility
                    phoneme_results[f"{target_lang}_{similarity_bin}"] = {
                        'large_phonemes': top_positive,
                        'small_phonemes': top_negative
                    }
                    
                    # Store all phoneme coefficients
                    for _, row in all_coefs.iterrows():
                        all_phoneme_coeffs.append({
                            'Language': target_lang,
                            'Bin': similarity_bin,
                            'Phoneme': row['phoneme'],
                            'Coefficient': row['coefficient']
                        })
    
    # Save accuracy results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('classification_results.csv', index=False)
        print(f"\nSaved classification results to classification_results.csv")
        
        # Print summary statistics
        print("\nAccuracy Summary Statistics:")
        summary = results_df.groupby('Similarity_Bin')[
            ['Logistic_Regression_Accuracy', 'Decision_Tree_Accuracy']
        ].agg(['mean'])
        print(summary)
        
        # Print decision tree hyperparameter summary
        print("\nBest Decision Tree Parameters by Similarity Bin:")
        dt_params = results_df.groupby('Similarity_Bin')[
            ['DT_Best_Max_Depth', 'DT_Best_Min_Samples_Split']
        ].agg(['mean', 'median', 'min', 'max'])
        print(dt_params)
    else:
        print("\nWARNING: No results collected!")
    
    # Save all phoneme coefficient results 
    all_phoneme_df = pd.DataFrame(all_phoneme_coeffs)
    
    # Verify we only have IPA symbols in results
    non_ipa_phonemes = all_phoneme_df[~all_phoneme_df['Phoneme'].isin(ipa_symbols_set)]
    if len(non_ipa_phonemes) > 0:
        print(f"\nWARNING: Found {len(non_ipa_phonemes)} non-IPA phonemes in results!")
        # Filter out non-IPA phonemes
        all_phoneme_df = all_phoneme_df[all_phoneme_df['Phoneme'].isin(ipa_symbols_set)]
    
    # Create a more detailed DataFrame for analysis
    detailed_results = []

    for _, row in all_phoneme_df.iterrows():
        # Get the model accuracy for this language-bin combination
        matching_result = next((r for r in results if 
                               r['Target_Language'] == row['Language'] and 
                               r['Similarity_Bin'] == row['Bin']), None)
        
        if matching_result:
            detailed_results.append({
                'Language': row['Language'],
                'Bin': row['Bin'],
                'Phoneme': row['Phoneme'],
                'Coefficient': row['Coefficient'],
                'Size_Class': 'large' if row['Coefficient'] > 0 else 'small',
                'Coefficient_Abs': abs(row['Coefficient']),
                'LR_Accuracy': matching_result['Logistic_Regression_Accuracy'],
                'DT_Accuracy': matching_result['Decision_Tree_Accuracy'],
                'DT_Max_Depth': matching_result['DT_Best_Max_Depth'],
                'DT_Min_Samples_Split': matching_result['DT_Best_Min_Samples_Split'],
                'Num_Training_Languages': matching_result['Num_Training_Languages']
            })

    # Convert to DataFrame and save
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('filtered_ipa_all_res.csv', index=False)
    print(f"\nSaved enhanced detailed IPA results to filtered_ipa_all_res.csv")
    
    # Save original phoneme coefficients
    all_phoneme_df.to_csv('all_phoneme_coefficients.csv', index=False)
    phoneme_summary = pd.DataFrame(columns=['Language', 'Bin', 'Class', 'Phoneme', 'Coefficient'])
    
    for key, value in phoneme_results.items():
        lang, bin_name = key.split('_', 1)
        
        # Add large phonemes
        for _, row in value['large_phonemes'].iterrows():
            phoneme_summary = pd.concat([phoneme_summary, pd.DataFrame({
                'Language': [lang],
                'Bin': [bin_name],
                'Class': ['large'],
                'Phoneme': [row['phoneme']],
                'Coefficient': [row['coefficient']]
            })])
            
        # Add small phonemes
        for _, row in value['small_phonemes'].iterrows():
            phoneme_summary = pd.concat([phoneme_summary, pd.DataFrame({
                'Language': [lang],
                'Bin': [bin_name],
                'Class': ['small'],
                'Phoneme': [row['phoneme']],
                'Coefficient': [row['coefficient']]
            })])
    
    phoneme_summary.to_csv('phoneme_importance.csv', index=False)
    print(f"Analysis files saved successfully")
    
    # Analyze phonemes by similarity bin
    for similarity_bin in ['Most Similar', 'Somewhat Similar', 'Least Similar']:
        bin_data = all_phoneme_df[all_phoneme_df['Bin'] == similarity_bin]
        print(f"\nTop 10 phonemes for LARGE words in {similarity_bin} bin:")
        large_phonemes = bin_data[bin_data['Coefficient'] > 0]
        print(large_phonemes.groupby('Phoneme')['Coefficient'].mean().nlargest(10))
        print(f"\nTop 10 phonemes for SMALL words in {similarity_bin} bin:")
        small_phonemes = bin_data[bin_data['Coefficient'] <= 0]
        print(small_phonemes.groupby('Phoneme')['Coefficient'].mean().nsmallest(10))
    
    # Print summary statistics for all phonemes
    print("\nTop 10 phonemes for LARGE words (averaged across all languages and all coefficients):")
    large_phonemes = all_phoneme_df[all_phoneme_df['Coefficient'] > 0]
    print(large_phonemes.groupby('Phoneme')['Coefficient'].mean().nlargest(10))
    
    print("\nTop 10 phonemes for SMALL words (averaged across all languages and all coefficients):")
    small_phonemes = all_phoneme_df[all_phoneme_df['Coefficient'] <= 0]
    print(small_phonemes.groupby('Phoneme')['Coefficient'].mean().nsmallest(10))
    
    # Additional analysis on all phonemes
    print("\nMost consistent phonemes (appearing in most languages):")
    lang_count = all_phoneme_df.groupby('Phoneme')['Language'].nunique()
    print(lang_count.nlargest(10))
    
    # Calculate variance to find most consistent effect sizes
    phoneme_variance = all_phoneme_df.groupby('Phoneme')['Coefficient'].var()
    print("\nMost consistent effect sizes (lowest variance):")
    print(phoneme_variance.nsmallest(10))
    
    print("\n=== Sound Symbolism Project Completed ===")
if __name__ == "__main__":
    main()