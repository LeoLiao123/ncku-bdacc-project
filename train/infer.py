import torch
import argparse
import pandas as pd
import os

from posatt_bilstm import PosAttBiLSTM
from train import preprocess_text, build_vocab

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = 2  # Truthful, Deceptive
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.4
DEFAULT_MAX_LEN = 500

def predict_custom(text_string, model, vocab, device, max_len=DEFAULT_MAX_LEN):
    """
    Predicts the class of a custom text string.
    """
    model.eval()

    processed_text = preprocess_text(text_string)

    tokens = processed_text.split()
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]

    if len(token_ids) < max_len:
        token_ids.extend([vocab['<PAD>']] * (max_len - len(token_ids)))
    else:
        token_ids = token_ids[:max_len]

    input_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction_index = torch.argmax(probabilities, dim=1).item()

    predicted_label = "Deceptive" if prediction_index == 1 else "Truthful"
    
    return predicted_label, probabilities.cpu().numpy()[0]

def main():
    parser = argparse.ArgumentParser(description='Infers custom text using a PosAtt-BiLSTM model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pt model file.')
    parser.add_argument('--text', type=str, required=True, help='Custom text string for inference.')
    parser.add_argument('--data_path', type=str, default='deceptive-opinion-merge-3.csv', 
                        help='Path to the dataset CSV for building the vocabulary.')
    parser.add_argument('--max_len', type=int, default=DEFAULT_MAX_LEN, help='Maximum sequence length for input text.')

    args = parser.parse_args()

    try:
        import nltk
        nltk.word_tokenize("test")
    except LookupError:
        print("NLTK 'punkt' resource not found.")
        print("Please execute the following in your Python environment:")
        print("import nltk")
        print("nltk.download('punkt')")
        return
    except Exception as e:
        print(f"Error loading NLTK: {e}")
        return


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading data from {args.data_path} to build vocabulary...")
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}. This file is required to build the vocabulary.")
        print("Please ensure the file exists or provide the correct path using --data_path.")
        return
    
    try:
        df = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"Error reading data file {args.data_path}: {e}")
        return

    texts_for_vocab = df['text'].apply(preprocess_text).values
    vocab = build_vocab(texts_for_vocab)
    print(f"Vocabulary size: {len(vocab)}")

    if '<PAD>' not in vocab or '<UNK>' not in vocab:
        print("Error: '<PAD>' or '<UNK>' token not found in vocabulary. Ensure build_vocab function handles this correctly.")
        return

    print("Initializing model...")
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

    print(f"Loading model state from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}.")
        return

    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch_info = checkpoint.get('epoch', 'Unknown')
            val_f1_info = checkpoint.get('val_f1', 'N/A')
            if val_f1_info != 'N/A':
                try:
                    val_f1_info = f"{float(val_f1_info):.4f}"
                except ValueError:
                    pass
            print(f"Model loaded from epoch {epoch_info}, validation F1: {val_f1_info}")
        else:
            model.load_state_dict(checkpoint)
            print("Model state_dict loaded directly.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the model architecture in this script matches the saved model.")
        print(f"Expected DROPOUT: {DROPOUT}. Update this script if the saved model uses a different dropout value.")
        return
        
    model.eval()
    args.text = """
    "From Knowledge to Wisdom

    Journal of Mechanics Engineering and Automation

    DOI: 10.17265/2159-5275   

    Print ISSN: 2159-5275 Online ISSN: 2159-5283 Frequency: bimonthly Current Volume: 13/2023

    Call for Papers and Books

    Dear 廖廷緯,

    Greetings from Journal of Mechanics Engineering and Automation, and hope this letter finds you well.

    We have heard that you submitted a paper '結合深度學習與影像處理之符咒圖像部件偵測與分割' from '臺灣網際網路研討會 (The 29th TANET 2023), 1-3 November 2023, Taiwan'. We are very interested in publishing some papers from you. If the paper mentioned has not been published in other journals and you have the idea of making our journal a vehicle for your research interests, you can send your paper in MS Word format to the following email or our online submission system. Or, you’d better send us your other unpublished papers or books.

    Hope to keep in touch and can publish some papers or books from you and your friends.

    Journal Description:

    Journal of Mechanics Engineering and Automation (JMEA), a professional journal published across the United States by David Publishing Company, USA. JMEA is a scholarly peer-reviewed international journal published monthly for educators and researchers in the relevant fields. It seeks to bridge and integrate the intellectual, methodological, and substantive diversity of scholarship, and to encourage a vigorous dialogue between scholars and practitioners. The journal welcomes contributions which promote the exchange of ideas and rational discourse between educators and researchers all over the world.

    JMEA is collected and indexed by the Library of U.S Congress, and it is also retrieved by some renowned databases:

    ★ Chinese Database of CEPS, Airiti Inc. & OCLC  
    ★ Chinese Scientific Journals Database, VIP Corporation, Chongqing, China  
    ★ Crossref  
    ★ Google Scholar  
    ★ Index Copernicus, Poland  
    ★ Norwegian Social Science Data Services (NSD), Norway  
    ★ PBN  
    ★ WorldCat  

    Information for Authors:

    1. The manuscript should be original and not have been published previously. Do not submit material that is currently being considered by another journal.  
    2. The manuscript should be written in English, and may be 3000–8000 words or longer if approved by the editor, including an abstract, text, tables, footnotes, appendixes, and references. The title should be on page 1 and not exceed 15 words, followed by an abstract of 100–200 words. 3–5 keywords or key phrases are required.  
    3. The manuscript should be in MS Word format; submit it to our email address.  
    4. Authors of accepted articles are required to sign the Transfer of Copyright Agreement form.  
    5. Authors will receive one hard copy of the issue containing their article.  
    6. Authors must pay USD 60 per page for publication.

    Submission:

    Send manuscripts online or via email to: mechanics@davidpublishing.com; JMEA-mechanics@hotmail.com; mechanics@davidpublisher.org.  
    (Click here for our automatic paper submission system.)

    Best Wishes!

    Xenia Chen  
    Editors Department  
    David Publishing Company  
    3 Germay Dr., Unit 4 #4651, Wilmington DE 19804  
    TEL: 001-323-984-7526; 001-323-410-1082 FAX: 001-323-984-7374

    If you do not wish to receive such emails in the future, please reply with “unsubscribe” in the subject line."

    """
    
    print(f"\nInferring text: \"{args.text}\"")
    predicted_label, probabilities = predict_custom(args.text, model, vocab, device, args.max_len)

    print(f"Predicted label: {predicted_label}")
    print(f"Probabilities (Truthful, Deceptive): [{probabilities[0]:.4f}, {probabilities[1]:.4f}]")

if __name__ == '__main__':
    main()
