# Deceptive Text Detection Project

## Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd ncku-bdacc-project
    ```

2.  **Set up a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install fastapi "uvicorn[standard]" pandas torch nltk python-dotenv
    ```

## Running the Application

1.  **Start the backend server:**

    ```bash
    cd backend
    python inference_api.py
    ```

2.  **Open the frontend:**

    Open the `frontend/index.html` file in your web browser.

## Usage

1.  **Enter text:**

    In the text area, type or paste the text you want to analyze.

2.  **Click "偵測垃圾郵件":**

    Click the button to send the text to the backend for analysis.

3.  **View the results:**

    The results, including the spam probability, a description of the spam likelihood, and the word count, will be displayed in the "結果" section.  A loading indicator will appear while the analysis is in progress.  Error messages will be displayed if any issues occur during detection.

## Example API Usage (for developers)

You can also directly interact with the API using `curl` or a similar tool:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "This is a positive review!"}' http://localhost:8000/detect_spam
```

This will return a JSON response containing the `spam_probability` and `word_count`.


## Notes

*   Ensure that the model file (`best_model.pt`) and data file (`deceptive-opinion-merge-3.csv`) are located in the `backend` directory, or adjust the `model_path` and `data_path` parameters in the `TextInput` model and the API calls accordingly.
*   The frontend JavaScript code assumes the backend is running on `http://localhost:8000`.  Modify the `backendUrl` variable in `frontend/script.js` if your backend is running elsewhere.
*   This project uses a pre-trained model.  For information on training the model, refer to the original research paper or the training scripts (if available).
