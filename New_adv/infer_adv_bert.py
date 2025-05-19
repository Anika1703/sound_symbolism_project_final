import os
import argparse
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from train_adv import AdversarialSoundSymbolismModel, IPADataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Adversarial BERT Sound Symbolism Model')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to trained adv model directory')
    parser.add_argument('--bert_model_name_or_path', type=str, required=True,
                        help='Same BERT path used during training')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save predictions & embeddings')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    df = pd.read_csv(args.test_data)
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name_or_path)
    test_ds = IPADataset(df, tokenizer, max_length=64)
    loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Load model
    model = AdversarialSoundSymbolismModel.load(args.model_dir).to(device)
    model.eval()

    all_true_symb, all_pred_symb = [], []
    all_true_lang, all_pred_lang = [], []
    all_embeddings = []
    all_texts = []

    with torch.no_grad():
        for inputs, symb_labels, lang_labels in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            symb_labels = symb_labels.to(device)
            lang_labels = lang_labels.to(device)

            embeddings = model.forward_embeddings(**inputs)
            symb_logits = model.size_classifier(embeddings)
            lang_logits = model.language_classifier(embeddings)

            symb_preds = torch.argmax(symb_logits, dim=1)
            lang_preds = torch.argmax(lang_logits, dim=1)

            all_true_symb.extend(symb_labels.cpu().tolist())
            all_pred_symb.extend(symb_preds.cpu().tolist())
            all_true_lang.extend(lang_labels.cpu().tolist())
            all_pred_lang.extend(lang_preds.cpu().tolist())
            all_embeddings.append(embeddings.cpu())
            # texts
            all_texts.extend(df['transcription'].tolist()[len(all_texts):len(all_texts)+len(symb_preds)])

    embeddings = torch.cat(all_embeddings, dim=0).numpy()

    # Metrics
    print("\n--- Sound Symbolism ---")
    print(f"Accuracy: {accuracy_score(all_true_symb, all_pred_symb):.4f}")
    print(classification_report(all_true_symb, all_pred_symb))

    print("\n--- Language Classification ---")
    print(f"Accuracy: {accuracy_score(all_true_lang, all_pred_lang):.4f}")
    print(classification_report(all_true_lang, all_pred_lang))

    # Save outputs
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_df = pd.DataFrame({
            'transcription': all_texts,
            'true_symbolism': all_true_symb,
            'pred_symbolism': all_pred_symb,
            'true_language': all_true_lang,
            'pred_language': all_pred_lang,
        })
        # append each embedding dim
        for i in range(embeddings.shape[1]):
            out_df[f'emb_{i}'] = embeddings[:, i]
        out_df.to_csv(os.path.join(args.output, 'predictions.csv'), index=False)
        print(f"Results saved to {os.path.join(args.output, 'predictions.csv')}")

if __name__ == '__main__':
    main()