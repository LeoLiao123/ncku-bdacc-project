document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    const detectButton = document.getElementById('detectButton');
    const spamPercentageSpan = document.getElementById('spamPercentage');
    const wordCountSpan = document.getElementById('wordCount');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorArea = document.getElementById('errorArea');
    const spamProbabilityDescription = document.getElementById('spamProbabilityDescription');

    // Backend API URL
    const backendUrl = 'http://localhost:8000/detect_spam';

    detectButton.addEventListener('click', async () => {
        const text = textInput.value;
        if (!text.trim()) {
            alert('Please enter text to analyze!');
            return;
        }

        // Clear previous results and errors
        spamPercentageSpan.textContent = '--';
        wordCountSpan.textContent = '--';
        errorArea.textContent = '';
        errorArea.style.display = 'none';
        loadingIndicator.style.display = 'block'; // Show loading indicator
        detectButton.disabled = true; // Disable button

        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            loadingIndicator.style.display = 'none'; // Hide loading indicator
            detectButton.disabled = false; // Enable button

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error response' }));
                throw new Error(`Server error: ${response.status} ${response.statusText}. ${errorData.detail || ''}`);
            }

            const result = await response.json();

            spamPercentageSpan.textContent = (result.spam_probability * 100).toFixed(2);
            wordCountSpan.textContent = result.word_count;

            // Determine spam probability
            let description = '';
            if (result.spam_probability >= 0.5) {
                description = 'High probability';
                spamPercentageSpan.style.color = 'red'; // Set probability to red
                spamPercentageSpan.style.fontWeight = 'bold'; // Bold
            } else {
                description = 'Safe';
                spamPercentageSpan.style.color = 'green'; // Set probability to green
                spamPercentageSpan.style.fontWeight = 'normal'; // Restore default weight
            }

            spamPercentageSpan.textContent = (result.spam_probability * 100).toFixed(2);
            spamProbabilityDescription.textContent = description;
            wordCountSpan.textContent = result.word_count;

        } catch (error) {
            console.error('Detection failed:', error);
            loadingIndicator.style.display = 'none'; // Hide loading indicator
            detectButton.disabled = false; // Enable button
            errorArea.textContent = `Detection failed: ${error.message}`;
            errorArea.style.display = 'block';
        }
    });

    // Optional: Live word count
    textInput.addEventListener('input', () => {
        const text = textInput.value;
        const words = text.trim().split(/\s+/).filter(word => word.length > 0);
        wordCountSpan.textContent = words.length; // Or update from backend response
    });
});
