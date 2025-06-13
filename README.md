# Deceptive Text Detection System

A web-based system for detecting deceptive text using a PosAtt-BiLSTM model with hybrid attention mechanism.

## Quick Start

Note : Please clone the repository and navigate to the project directory.

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000
```

## Usage

1.  Open `frontend/index.html`
2.  Enter text in the input area
3.  Click "Detect Deceptive Text"
4.  View results including:
    *   Deceptive probability percentage
    *   Suspicious keywords with attention scores
    *   Text visualization with attention highlighting

## API

Backend API available at `http://localhost:8000/docs` for direct integration.

POST `/detect_spam`:

```json
{
  "text": "Your text here"
}
```

Returns deceptive probability, word analysis, and attention weights.
