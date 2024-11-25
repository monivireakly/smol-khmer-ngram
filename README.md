# Khmer Syllable Prediction
A simple n-gram based model for predicting the next character in Khmer text sequences.

### Features
- Character-level prediction for Khmer text
- Supports MPS (Apple Silicon), CUDA, and CPU
- Temperature-based sampling
- Sparsity control

### Training a New Model
```python
git clone https://github.com/monivireakly/smol-khmer-ngram.git
cd khmer-syllable-prediction
pip install -r requirements.txt

from subword import KhmerSyllableTokenizer, get_khmer_text, NGramConfig, build_ngram_model

# Load your data
data = get_khmer_text()

# Initialize tokenizer
tokenizer = KhmerSyllableTokenizer()
tokenizer.fit(data['train'])

# Configure training
config = NGramConfig()
config.epochs = 300
config.learning_rate = 0.05
config.initial_temp = 1
config.temp_decay = 0.9
config.alpha = 0.3

# Train model
trans_matrix, bias, history = build_ngram_model(tokenizer, data['train'], config)
```

### Making Predictions
```python
from subword import predict_next_syllables

# Example text
text = "កម្ពុជ"  # Input text without last character

# Get predictions
predictions = predict_next_syllables(text, tokenizer, trans_matrix, bias, k=5)

# Print predictions
for char, prob in predictions:
    print(f"{char} -> {prob:.3f}")
```

### Testing the Model
```python
from test_model import test_prediction

# Test single example
text = "សួស្តី"
test_prediction(text, tokenizer, trans_matrix, bias)
```

### Model Files
The trained models are saved in the models/ directory with the format:
- Model weights: ngram_model_[timestamp].pt
- Training history: training_history_[timestamp].json
