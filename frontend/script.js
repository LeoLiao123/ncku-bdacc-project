document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    const detectButton = document.getElementById('detectButton');
    const spamPercentageSpan = document.getElementById('spamPercentage');
    const wordCountSpan = document.getElementById('wordCount');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorArea = document.getElementById('errorArea');
    const spamProbabilityDescription = document.getElementById('spamProbabilityDescription');
    const keywordsSection = document.getElementById('keywordsSection');
    const keywordsList = document.getElementById('keywordsList');
    const textVisualization = document.getElementById('textVisualization');
    const highlightedText = document.getElementById('highlightedText');

    // Backend API URL
    const backendUrl = 'http://localhost:8000/detect_spam';

    function displaySuspiciousKeywords(keywords) {
        if (keywords && keywords.length > 0) {
            keywordsList.innerHTML = '';
            keywords.forEach(keywordObj => {
                const word = Object.keys(keywordObj)[0];
                const score = Object.values(keywordObj)[0];
                
                const keywordElement = document.createElement('span');
                keywordElement.className = 'keyword-tag';
                keywordElement.textContent = `${word} (${(score * 100).toFixed(1)}%)`;
                
                // Color intensity based on score
                const intensity = Math.min(score * 2, 1); // Scale for better visibility
                keywordElement.style.backgroundColor = `rgba(255, 99, 71, ${intensity})`;
                keywordElement.style.color = intensity > 0.5 ? 'white' : 'black';
                
                keywordsList.appendChild(keywordElement);
            });
            keywordsSection.style.display = 'block';
        } else {
            keywordsSection.style.display = 'none';
        }
    }

    function highlightTextWithAttention(processedTokens, attentionWeights, attentionThreshold = 0.3) {
        if (!processedTokens || processedTokens.length === 0) {
            textVisualization.style.display = 'none';
            return;
        }

        let highlightedHTML = '';
        for (let i = 0; i < processedTokens.length; i++) {
            const word = processedTokens[i];
            let backgroundColor = 'transparent';
            
            if (attentionWeights && i < attentionWeights.length) {
                const weight = attentionWeights[i];
                // Ensure weight is a number and not NaN
                const validWeight = (typeof weight === 'number' && !isNaN(weight)) ? weight : 0;
                
                // Only highlight if this specific token's attention weight exceeds threshold
                if (validWeight > attentionThreshold) {
                    const intensity = Math.min(validWeight, 1); 
                    backgroundColor = `rgba(255, 99, 71, ${intensity * 0.7})`;
                }
            }
            
            highlightedHTML += `<span class="attention-word" style="background-color: ${backgroundColor}; padding: 2px 4px; margin: 1px; border-radius: 3px;">${word}</span> `;
        }
        
        highlightedText.innerHTML = highlightedHTML;
        textVisualization.style.display = 'block';
    }

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
        keywordsSection.style.display = 'none';
        textVisualization.style.display = 'none';
        loadingIndicator.style.display = 'block';
        detectButton.disabled = true;

        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            loadingIndicator.style.display = 'none';
            detectButton.disabled = false;

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
                spamPercentageSpan.style.color = 'red';
                spamPercentageSpan.style.fontWeight = 'bold';
            } else {
                description = 'Safe';
                spamPercentageSpan.style.color = 'green';
                spamPercentageSpan.style.fontWeight = 'normal';
            }

            spamProbabilityDescription.textContent = description;

            // Display suspicious keywords
            displaySuspiciousKeywords(result.suspicious_keywords);
            
            // Always show text visualization if we have processed tokens
            if (result.processed_tokens_for_attention && result.processed_tokens_for_attention.length > 0) {
                // Pass attention threshold (0.3) - only tokens with attention > 0.3 will be highlighted
                highlightTextWithAttention(result.processed_tokens_for_attention, result.attention_weights, 0.8);
            } else {
                textVisualization.style.display = 'none';
            }

        } catch (error) {
            console.error('Detection failed:', error);
            loadingIndicator.style.display = 'none';
            detectButton.disabled = false;
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