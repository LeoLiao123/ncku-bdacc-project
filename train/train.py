import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from posatt_bilstm import PosAttBiLSTM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import argparse
import ssl
import time
import os
from gensim.models import KeyedVectors, Word2Vec
import gensim.downloader as api
import matplotlib.pyplot as plt
from tqdm import tqdm
from focal_loss import FocalLoss

# Download required NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=500):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = text.split()
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        if len(indices) < self.max_len:
            indices = indices + [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
            
        return torch.tensor(indices), torch.tensor(label)

def preprocess_text(text):
    text = text.lower()
    
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    tokens = word_tokenize(text)
    
    important_stopwords = {'not', 'no', 'never', 'nothing'}
    stop_words = set(stopwords.words('english')) - important_stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def build_vocab(texts, min_freq=2):
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
            
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
            
    return vocab

def load_word_vectors(vocab, texts, vector_size=300):
    """
    Load pre-trained word vectors or train CBOW model from scratch
    """
    if os.path.exists('word2vec_model.bin'):
        print("Loading previously trained word vectors...")
        model = Word2Vec.load('word2vec_model.bin')
        
        weights_matrix = np.zeros((len(vocab), vector_size))
        words_found = 0
        
        for word, idx in vocab.items():
            try:
                weights_matrix[idx] = model.wv[word]
                words_found += 1
            except KeyError:
                weights_matrix[idx] = np.random.normal(scale=0.6, size=(vector_size,))
        
        print(f"Loaded {words_found} word vectors from saved model")
        return weights_matrix
    
    try:
        print("Trying to load pre-trained word vectors...")
        try:
            word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            print("Successfully loaded pre-trained word vectors from local")
        except FileNotFoundError:
            print("Pre-trained word vectors not found locally, trying to download automatically using gensim.downloader...")
            import gensim.downloader as api
            model_path = api.load("word2vec-google-news-300", return_path=True)
            print(f"Model downloaded to: {model_path}")
            
            word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
            print("Successfully loaded pre-trained word vectors")
        
        weights_matrix = np.zeros((len(vocab), vector_size))
        words_found = 0
        
        for word, idx in vocab.items():
            try:
                weights_matrix[idx] = word_vectors[word]
                words_found += 1
            except KeyError:
                weights_matrix[idx] = np.random.normal(scale=0.6, size=(vector_size,))
        
        print(f"Loaded {words_found} word vectors from pre-trained model")
        return weights_matrix
    
    except Exception as e:
        print(f"Failed to load pre-trained word vectors: {e}")
        print("Training CBOW model from scratch...")
        tokenized_texts = [text.split() for text in texts]
        
        model = Word2Vec(
            tokenized_texts,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=4,
            sg=0
        )
        
        model.save('word2vec_model.bin')
        print("Saved trained CBOW model for future use")
        
        weights_matrix = np.zeros((len(vocab), vector_size))
        words_found = 0
        
        for word, idx in vocab.items():
            try:
                weights_matrix[idx] = model.wv[word]
                words_found += 1
            except KeyError:
                weights_matrix[idx] = np.random.normal(scale=0.6, size=(vector_size,))
        
        print(f"Trained CBOW model with {words_found} words")
        return weights_matrix

def train_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs=20, scheduler=None):
    best_val_f1 = 0
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
            
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        val_precision = precision_score(val_targets, val_preds, average='weighted')
        val_recall = recall_score(val_targets, val_preds, average='weighted')
        
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        
        print(f'Epoch: {epoch+1}')
        print(f'Average Loss: {avg_train_loss:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        print(f'Validation F1: {val_f1:.4f}')
        print(f'Validation Precision: {val_precision:.4f}')
        print(f'Validation Recall: {val_recall:.4f}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc
            }, 'best_model.pt')
            print(f'New best model saved with F1 score: {val_f1:.4f}')
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Accuracy')
    plt.plot(val_f1_scores, label='F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    
    return train_losses, val_accuracies, val_f1_scores

def test_model(model, test_loader, device):
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='weighted')
    test_precision = precision_score(test_targets, test_preds, average='weighted')
    test_recall = recall_score(test_targets, test_preds, average='weighted')
    
    print("\nTest Results:")
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=['Truthful', 'Deceptive']))
    
    return test_acc, test_f1, test_precision, test_recall

def analyze_attention_weights(model, test_loader, device, vocab_inverse, n_examples=5):
    """
    Visualize and analyze attention weights for a few examples
    """
    model.eval()
    
    PAD_IDX = 0
    
    examples = []
    with torch.no_grad():
        for data, target in test_loader:
            if len(examples) >= n_examples:
                break
                
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            pred = output.argmax(dim=1)
            
            for i in range(min(len(data), n_examples - len(examples))):
                tokens = [vocab_inverse[idx.item()] for idx in data[i] if idx.item() != PAD_IDX]
                true_label = target[i].item()
                pred_label = pred[i].item()
                examples.append({
                    'tokens': tokens,
                    'true_label': true_label,
                    'pred_label': pred_label
                })
    
    print("\nExample predictions:")
    for i, example in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Text: {' '.join(example['tokens'][:20])}...")
        print(f"True label: {'Deceptive' if example['true_label'] == 1 else 'Truthful'}")
        print(f"Predicted label: {'Deceptive' if example['pred_label'] == 1 else 'Truthful'}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Train and test PosAtt-BiLSTM model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'analyze'],
                      help='Mode: train, test, or analyze')
    parser.add_argument('--data_path', type=str, default='deceptive-opinion-merge-3.csv',
                      help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Word embedding dimension')
    parser.add_argument('--max_len', type=int, default=500, help='Maximum sequence length')
    args = parser.parse_args()

    EMBEDDING_DIM = args.embedding_dim
    HIDDEN_DIM = args.hidden_dim
    OUTPUT_DIM = 2
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = args.dropout
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    MAX_LEN = args.max_len
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading and preprocessing data...')
    df = pd.read_csv(args.data_path)
    
    texts = df['text'].apply(preprocess_text).values
    labels = (df['deceptive'] == 'deceptive').astype(int).values
    
    vocab = build_vocab(texts)
    print(f'Vocabulary size: {len(vocab)}')
    
    vocab_inverse = {idx: word for word, idx in vocab.items()}
    
    weights_matrix = load_word_vectors(vocab, texts, vector_size=EMBEDDING_DIM)
    
    if "Spam" in args.data_path:
        test_size = 0.2
    elif "Yelp" in args.data_path:
        test_size = 0.1
    else:
        test_size = 0.2
        
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    if "Yelp" in args.data_path:
        val_size = 0.1
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
    else:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
        )
        test_texts, test_labels = temp_texts, temp_labels
    
    if args.mode == 'train':
        print(f'Train set: {len(train_texts)} samples')
        print(f'Validation set: {len(val_texts)} samples')
        print(f'Test set: {len(test_texts)} samples')
        
        train_dataset = ReviewDataset(train_texts, train_labels, vocab, MAX_LEN)
        val_dataset = ReviewDataset(val_texts, val_labels, vocab, MAX_LEN)
        test_dataset = ReviewDataset(test_texts, test_labels, vocab, MAX_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        model = PosAttBiLSTM(
            vocab_size=len(vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT,
            pad_idx=vocab['<PAD>']
        ).to(device)
        
        model.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5.778535298779227e-06)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        
        criterion = FocalLoss(alpha=0.9, gamma=3.0)
        
        print('Starting training...')
        start_time = time.time()
        train_losses, val_accuracies, val_f1_scores = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, n_epochs=args.epochs, scheduler=scheduler
        )
        training_time = time.time() - start_time
        print(f'Training completed in {training_time:.2f} seconds')
        
        print('\nTesting on test set...')
        checkpoint = torch.load('best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = test_model(model, test_loader, device)
        
        print('\nAnalyzing attention weights...')
        analyze_attention_weights(model, test_loader, device, vocab_inverse)
        
    elif args.mode == 'test':
        test_dataset = ReviewDataset(texts, labels, vocab, MAX_LEN)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        model = PosAttBiLSTM(
            vocab_size=len(vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT,
            pad_idx=vocab['<PAD>']
        ).to(device)
        
        checkpoint = torch.load('best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded trained model from epoch {checkpoint["epoch"]} with validation F1: {checkpoint["val_f1"]:.4f}')
        
        test_metrics = test_model(model, test_loader, device)
        
    elif args.mode == 'analyze':
        subset_size = min(1000, len(texts))
        subset_texts = texts[:subset_size]
        subset_labels = labels[:subset_size]
        
        test_dataset = ReviewDataset(subset_texts, subset_labels, vocab, MAX_LEN)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        model = PosAttBiLSTM(
            vocab_size=len(vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT,
            pad_idx=vocab['<PAD>']
        ).to(device)
        
        checkpoint = torch.load('best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded trained model from epoch {checkpoint["epoch"]} with validation F1: {checkpoint["val_f1"]:.4f}')
        
        analyze_attention_weights(model, test_loader, device, vocab_inverse, n_examples=10)

if __name__ == '__main__':
    main()
