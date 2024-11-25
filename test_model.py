import torch
from subword import KhmerSyllableTokenizer, predict_next_syllables, get_khmer_text
import os
import random

# Check for MPS (Apple Silicon) availability
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint['trans_matrix'].to(device), checkpoint['bias'].to(device), checkpoint['vocab']

def test_prediction(text, tokenizer, trans_matrix, bias, k=5):
    input_text = text[:-1]
    actual_char = text[-1]
    
    predictions = predict_next_syllables(input_text, tokenizer, trans_matrix, bias, k=k)
    
    print("\nTest Results:")
    print(f"Input text: {input_text}")
    print(f"Actual next char: {actual_char}")
    print("\nModel predictions:")
    for char, prob in predictions:
        print(f"{char} -> {prob:.3f}" + (" ✓" if char == actual_char else ""))
    
    return actual_char in [p[0] for p in predictions]

if __name__ == "__main__":
    try:
        # Load data and get random samples
        data = get_khmer_text()
        test_examples = random.sample(data['test'], 1000)  # Get 100 random samples
        # test_examples = ["កម្ពុជា", "ឡើង", "ការលើកឡើង"]
        # Load latest model
        model_path = "models/ngram_model_20241121_1559.pt"
        trans_matrix, bias, vocab = load_model(model_path)
        
        # Initialize tokenizer with saved vocabulary
        tokenizer = KhmerSyllableTokenizer()
        tokenizer.str2idx = vocab
        tokenizer.idx2str = {v: k for k, v in vocab.items()}
        tokenizer.vocab_size = len(vocab)
        
        print(f"Using device: {device}")
        print(f"\nTesting predictions on {len(test_examples)} random examples:")
        print("=" * 50)
        
        correct = 0
        for example in test_examples:
            is_correct = test_prediction(example, tokenizer, trans_matrix, bias)
            correct += int(is_correct)
            print("-" * 50)
        
        print(f"\nAccuracy on {len(test_examples)} random test examples: {correct/len(test_examples):.2%}")
    
    except Exception as e:
        print(f"Error: {str(e)}") 