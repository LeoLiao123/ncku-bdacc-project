import torch
import pandas as pd
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple
import ssl

# Import model definition from posatt_bilstm.py
try:
    from posatt_bilstm import PosAttBiLSTM
except ImportError:
    raise ImportError("Could not import PosAttBiLSTM from posatt_bilstm.py. Ensure the file exists and is in the correct path.")

# --- NLTK Configuration ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' resource...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'stopwords' resource...")
    nltk.download('stopwords', quiet=True)

# --- Model and Training Hyperparameters ---
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = 2
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.4
DEFAULT_MAX_LEN = 500

# --- Global Variables ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Cache loaded models and vocabularies: key is (model_path, data_path)
loaded_models_vocabs: Dict[Tuple[str, str], Tuple[PosAttBiLSTM, Dict[str, int]]] = {}

# --- Functions copied from train.py ---
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    
    important_stopwords = {'not', 'no', 'never', 'nothing', "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't"}
    stop_words = set(stopwords.words('english')) - important_stopwords
    
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return " ".join(tokens)

def build_vocab(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    word_freq: Dict[str, int] = {}
    for text_content in texts:
        for word in text_content.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# --- FastAPI Application ---
app = FastAPI(title="Deceptive Text Detection API")
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str
    model_path: str = 'best_model.pt'  # Default model path
    data_path: str = 'deceptive-opinion-merge-new-2.csv'  # Default data path for vocabulary

class PredictionOutput(BaseModel):
    spam_probability: float
    word_count: int
    suspicious_keywords: List[Dict[str, float]]  # List of {word: score}
    attention_weights: List[float]  # Attention weights for each word
    processed_tokens_for_attention: List[str]  # Tokens corresponding to attention_weights

def extract_suspicious_keywords(tokens: List[str], attention_weights: List[float], threshold: float = 0.1) -> List[Dict[str, float]]:
    """Extract keywords with high attention weights as suspicious."""
    if len(tokens) != len(attention_weights):
        # Pad or truncate to match
        min_len = min(len(tokens), len(attention_weights))
        tokens = tokens[:min_len]
        attention_weights = attention_weights[:min_len]
    
    # Create list of word-weight pairs
    word_weights = []
    for token, weight in zip(tokens, attention_weights):
        if weight > threshold and token not in ['<PAD>', '<UNK>']:
            word_weights.append({token: float(weight)})
    
    # Sort by weight (descending)
    word_weights.sort(key=lambda x: list(x.values())[0], reverse=True)
    
    return word_weights[:10]  # Return top 10 suspicious words

def _load_model_and_vocab_from_path(model_path: str, data_path: str) -> Tuple[PosAttBiLSTM, Dict[str, int]]:
    """Load model and vocabulary from file paths."""
    global device, loaded_models_vocabs

    cache_key = (model_path, data_path)
    if cache_key in loaded_models_vocabs:
        print(f"Using cached model/vocab for {cache_key}")
        return loaded_models_vocabs[cache_key]

    print(f"Loading vocabulary from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file for vocab not found: {data_path}")
    try:
        df = pd.read_csv(data_path)
        if 'text' not in df.columns:
            raise ValueError(f"'{data_path}' 中找不到 'text' 欄位。")
    except Exception as e:
        raise RuntimeError(f"Error reading data file {data_path}: {e}")
    
    texts_for_vocab = df['text'].apply(preprocess_text).tolist()
    current_vocab = build_vocab(texts_for_vocab)
    print(f"Vocabulary size: {len(current_vocab)}")
    if '<PAD>' not in current_vocab or '<UNK>' not in current_vocab:
        raise ValueError("'<PAD>' or '<UNK>' missing from vocabulary.")

    print(f"Initializing model structure (importing PosAttBiLSTM)...")
    current_model = PosAttBiLSTM(
        vocab_size=len(current_vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_idx=current_vocab['<PAD>']
    ).to(device)

    print(f"Loading model state from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            current_model.load_state_dict(checkpoint['model_state_dict'])
            epoch_info = checkpoint.get('epoch', 'Unknown')
            val_f1_info = checkpoint.get('val_f1', 'N/A')
            if val_f1_info != 'N/A':
                try: val_f1_info = f"{float(val_f1_info):.4f}"
                except ValueError: pass
            print(f"Model loaded from epoch {epoch_info}, validation F1: {val_f1_info}")
        else:
            current_model.load_state_dict(checkpoint)
            print("Model state_dict loaded directly.")
    except Exception as e:
        raise RuntimeError(f"Error loading model state_dict from {model_path}: {e}. "
                           f"Ensure DROPOUT ({DROPOUT}) and HIDDEN_DIM ({HIDDEN_DIM}) in this script "
                           f"match the trained model and the PosAttBiLSTM definition in posatt_bilstm.py.")
    
    current_model.eval()  # Set to evaluation mode
    loaded_models_vocabs[cache_key] = (current_model, current_vocab)
    return current_model, current_vocab

@app.on_event("startup")
async def startup_event():
    """Preload the default model on application startup."""
    default_model_path = 'best_model.pt'
    default_data_path = 'deceptive-opinion-merge-new-2.csv'
    print("FastAPI application startup...")
    print(f"Attempting to preload model from: {default_model_path} and vocab from: {default_data_path}")
    try:
        _load_model_and_vocab_from_path(default_model_path, default_data_path)
        print(f"Successfully preloaded model and vocab for ({default_model_path}, {default_data_path})")
    except Exception as e:
        print(f"Could not preload default model/vocab: {e}")
        print("Model and vocab will be loaded on first valid request.")


@app.post("/detect_spam", response_model=PredictionOutput)
async def detect_spam(item: TextInput):
    if not item.text.strip():
        return PredictionOutput(
            spam_probability=0.0, 
            word_count=0, 
            suspicious_keywords=[], 
            attention_weights=[],
            processed_tokens_for_attention=[]
        )
    
    word_count = len(item.text.split())
    
    global device
    try:
        current_model, current_vocab = _load_model_and_vocab_from_path(item.model_path, item.data_path)
        
        current_model.eval()
        processed_text_str = preprocess_text(item.text)
        tokens = processed_text_str.split()
        print(f"Processed text tokens: {tokens} (count: {len(tokens)})")
        original_tokens_for_attention = tokens.copy()
        token_ids = [current_vocab.get(token, current_vocab['<UNK>']) for token in tokens]

        # Pad or truncate sequence
        if len(token_ids) < DEFAULT_MAX_LEN:
            # Pad tokens list as well
            padded_tokens = tokens + ['<PAD>'] * (DEFAULT_MAX_LEN - len(token_ids))
            token_ids.extend([current_vocab['<PAD>']] * (DEFAULT_MAX_LEN - len(token_ids)))
        else:
            padded_tokens = tokens[:DEFAULT_MAX_LEN]
            token_ids = token_ids[:DEFAULT_MAX_LEN]

        input_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            # Forward pass
            output = current_model(input_tensor)
            probabilities_tensor = torch.softmax(output, dim=1)
            prediction_index = torch.argmax(probabilities_tensor, dim=1).item()

            # Extract attention weights from model
            # We need to modify the forward pass to also return attention weights
            embedded = current_model.embedding(input_tensor)
            embedded = current_model.pos_encoding(embedded)
            embedded = current_model.dropout(embedded)
            
            lstm_output, _ = current_model.bilstm(embedded)
            reduced_output = current_model.dim_reduction(lstm_output)
            attended = current_model.hybrid_attention(reduced_output)
            
            # Calculate attention weights (sum across hidden dimensions)
            attention_weights = torch.mean(torch.abs(attended), dim=2).squeeze(0).cpu().numpy()
            
            # Normalize attention weights
            if attention_weights.max() > 0:
                attention_weights = attention_weights / attention_weights.max()

        predicted_label = "Deceptive" if prediction_index == 1 else "Truthful"
        probabilities_list = probabilities_tensor.cpu().numpy()[0].tolist()
        
        # Extract suspicious keywords
        suspicious_keywords = extract_suspicious_keywords(
            padded_tokens, 
            attention_weights.tolist(), 
            threshold=0.9
        )
        
        print(f"Suspicious Keywords: {suspicious_keywords}")
        
        final_attention_weights = attention_weights[:len(original_tokens_for_attention)].tolist()
        
        return PredictionOutput(
            spam_probability=probabilities_list[1], 
            word_count=word_count,
            suspicious_keywords=suspicious_keywords,
            attention_weights=final_attention_weights,
            processed_tokens_for_attention=original_tokens_for_attention
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Unexpected error during prediction: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {type(e).__name__}")

# Usage:
# 1. Save this code as fastapi_inference_app.py.
# 2. Ensure posatt_bilstm.py is in the same directory or in the Python path.
# 3. Install dependencies: pip install fastapi "uvicorn[standard]" pandas torch nltk
# 4. Run the app: uvicorn fastapi_inference_app:app --reload
# 5. Send POST requests to http://127.0.0.1:8000/detect_spam with a JSON body like:
#    {
#        "text": "This is a wonderful product, highly recommended!",
#        "model_path": "best_model.pt",
#        "data_path": "deceptive-opinion-merge-new-2.csv"
#    }
#    Or just the text (default paths will be used):
#    {
#        "text": "This is a wonderful product, highly recommended!"
#    }
# 6. See auto-generated API docs at http://127.0.0.1:8000/docs

if __name__ == "__main__":
    print("To run this application, use: uvicorn inference_api:app --reload")
    print(f"Using device: {device}")
    print("To run this application, use Uvicorn: uvicorn fastapi_inference_app:app --reload")
    print(f"Using device: {device}")
