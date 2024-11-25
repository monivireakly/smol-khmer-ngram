import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import json
import os
from datetime import datetime

# Check for MPS (Apple Silicon) availability
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

def compute_validation_loss(val_texts, trans_matrix, tokenizer):
    val_loss = 0
    for text in val_texts:
        indices = tokenizer.text_to_tensor(text)
        if len(indices) < 2:
            continue
        for i in range(len(indices) - 1):
            prev_idx, next_idx = indices[i], indices[i+1]
            logits = trans_matrix[prev_idx]
            loss = F.cross_entropy(logits.unsqueeze(0), next_idx.unsqueeze(0))
            val_loss += loss
    return val_loss.item()

class KhmerSyllableTokenizer:
    def __init__(self, min_freq=2):
        self.str2idx = {'<UNK>': 0, '<PAD>': 1}
        self.idx2str = {0: '<UNK>', 1: '<PAD>'}
        self.vocab_size = 2
        self.min_freq = min_freq
        self.char_counts = defaultdict(int)

    def split_khmer_chars(self, text):
        """Split text into individual Khmer characters"""
        chars = []
        for char in text:
            if '\u1780' <= char <= '\u17FF':  # Khmer Unicode range
                chars.append(char)
        
        # Print debug info only for the first text
        if not hasattr(self, '_debug_printed'):
            print(f"\nExample character splitting:")
            print(f"Text: {text} → Chars: {chars}")
            self._debug_printed = True
            
        return chars

    def fit(self, texts):
        # Count individual characters
        all_chars = []
        for text in texts:
            chars = self.split_khmer_chars(text)
            all_chars.extend(chars)
            for char in chars:
                self.char_counts[char] += 1
        
        # Print statistics
        print("\nVocabulary Statistics:")
        print(f"Total characters found: {len(all_chars)}")
        print(f"Unique characters: {len(self.char_counts)}")
        print("\nTop 10 most common characters:")
        for char, count in sorted(self.char_counts.items(), 
                                key=lambda x: x[1], reverse=True)[:10]:
            print(f"{char}: {count}")
        
        # Build vocabulary with frequent characters
        for char, count in self.char_counts.items():
            if count >= self.min_freq:
                self.str2idx[char] = self.vocab_size
                self.idx2str[self.vocab_size] = char
                self.vocab_size += 1
        
        print(f"\nFiltered vocabulary size: {self.vocab_size}")

    def text_to_tensor(self, text):
        chars = self.split_khmer_chars(text)
        indices = []
        for char in chars:
            if char in self.str2idx:
                indices.append(self.str2idx[char])
            else:
                indices.append(self.str2idx['<UNK>'])
        return torch.tensor(indices)

class NGramConfig:
    def __init__(self):
        # Data preprocessing
        self.min_freq = 2          # Minimum frequency for syllable inclusion
        self.val_split = 0.2       # Validation split ratio
        
        # Model initialization
        self.alpha = 0.1           # Initial value scaling
        self.hidden_size = 64      # Size of hidden projection (if needed)
        
        # Training parameters
        self.epochs = 20           # Number of training epochs
        self.batch_size = 32       # Batch size (if implemented)
        self.learning_rate = 0.01  # Initial learning rate
        self.min_lr = 0.001       # Minimum learning rate
        self.weight_decay = 1e-4   # L2 regularization
        
        # Temperature scheduling
        self.initial_temp = 1.0    # Initial softmax temperature
        self.temp_decay = 0.9      # Temperature decay rate
        self.min_temp = 0.5       # Minimum temperature
        
        # Early stopping
        self.patience = 5          # Early stopping patience
        self.min_delta = 1e-4      # Minimum change for improvement
        
        # Prediction
        self.pred_temp = 0.8       # Prediction temperature
        self.min_prob = 0.001      # Minimum probability threshold
        self.top_k = 2             # Number of top predictions

def build_ngram_model(tokenizer, texts, config=None):
    if config is None:
        config = NGramConfig()
    
    split_idx = int(len(texts) * (1 - config.val_split))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Model setup
    trans_matrix = nn.Parameter(torch.rand((tokenizer.vocab_size, tokenizer.vocab_size)) * config.alpha)
    bias = nn.Parameter(torch.zeros(tokenizer.vocab_size))
    
    class NGramModel(nn.Module):
        def __init__(self, trans_matrix, bias):
            super().__init__()
            self.trans_matrix = trans_matrix
            self.bias = bias
        
        def forward(self, temp=1.0):
            logits = self.trans_matrix + self.bias.unsqueeze(0)
            return torch.softmax(logits / temp, dim=1)
    
    model = NGramModel(trans_matrix, bias)
    # model = model # ve model to device
    optimizer = AdamW(model.parameters(), 
                     lr=config.learning_rate, 
                     weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, 
                                 T_max=config.epochs, 
                                 eta_min=config.min_lr)
    
    # Training tracking
    best_val_loss = float('inf')
    best_matrix = None
    best_bias = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'temp': [], 'sparsity': []}
    
    print("\nTraining Progress:")
    print("=" * 70)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Sparsity':>8} | {'LR':>10} | {'Temp':>8}")
    print("-" * 70)
    
    for epoch in range(config.epochs):
        # Temperature scheduling
        temp = max(config.initial_temp * (config.temp_decay ** epoch), config.min_temp)
        
        optimizer.zero_grad()
        probs = model(temp)
        
        # Training
        train_loss = torch.tensor(0.0, requires_grad=True)
        n_train = 0
        
        for text in train_texts:
            indices = tokenizer.text_to_tensor(text)
            if len(indices) < 2:
                continue
            for i in range(len(indices)-1):
                prev_idx = indices[i]
                next_idx = indices[i+1]
                train_loss = train_loss + (-torch.log(probs[prev_idx, next_idx] + 1e-8))
                n_train += 1
        
        if n_train > 0:
            train_loss = train_loss / n_train
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        with torch.no_grad():
            val_loss = 0
            n_val = 0
            for text in val_texts:
                indices = tokenizer.text_to_tensor(text)
                if len(indices) < 2:
                    continue
                for i in range(len(indices)-1):
                    prev_idx = indices[i]
                    next_idx = indices[i+1]
                    val_loss -= torch.log(probs[prev_idx, next_idx] + 1e-8)
                    n_val += 1
            
            if n_val > 0:
                val_loss = val_loss / n_val
        
        # Compute sparsity
        sparsity = 100 * (probs > config.min_prob).sum().item() / (tokenizer.vocab_size ** 2)
        
        # Update scheduler
        scheduler.step()
        
        # Track metrics
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        history['lr'].append(scheduler.get_last_lr()[0])
        history['temp'].append(temp)
        history['sparsity'].append(sparsity)
        
        print(f"{epoch+1:5d} | {train_loss.item():10.4f} | {val_loss.item():10.4f} | "
              f"{sparsity:8.2f} | {scheduler.get_last_lr()[0]:10.6f} | {temp:8.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_matrix = probs.detach().clone()
            best_bias = model.bias.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("\nEarly stopping triggered!")
                break
    
    # Print final statistics
    print("\nTraining Summary:")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final learning rate: {scheduler.get_last_lr()[0]:.6f}")
    print(f"Final temperature: {temp:.4f}")
    print(f"Final sparsity: {sparsity:.2f}%")
    
    # Save model and training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(save_dir, f"ngram_model_{timestamp}.pt")
    torch.save({
        'trans_matrix': best_matrix,
        'bias': best_bias,
        'config': vars(config),
        'vocab': tokenizer.str2idx
    }, model_path)
    
    # Save training history
    history_path = os.path.join(save_dir, f"training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"History saved to: {history_path}")
    
    return best_matrix, best_bias, history

def predict_next_syllables(text, tokenizer, trans_matrix, bias, k=5, temperature=0.8):
    indices = tokenizer.text_to_tensor(text)
    if len(indices) == 0:
        return []
    
    # Get last character index
    last_idx = indices[-1].item()  # Convert tensor to integer
    
    # Get probabilities
    logits = trans_matrix + bias.unsqueeze(0)
    probs = torch.softmax(logits[last_idx] / temperature, dim=0)
    
    # Filter and normalize
    min_prob_threshold = 0.01
    probs[probs < min_prob_threshold] = 0
    probs = probs / (probs.sum() + 1e-8)
    
    # Get top k
    top_k = torch.topk(probs, min(k, (probs > 0).sum().item()))
    
    print("\nPrediction Debug:")
    print(f"Input text: {text}")
    print(f"Characters: {tokenizer.split_khmer_chars(text)}")
    print(f"Indices: {indices}")
    print(f"Last character: {tokenizer.idx2str[last_idx]}")
    
    predictions = []
    for idx, prob in zip(top_k.indices, top_k.values):
        if prob.item() > 0:
            char = tokenizer.idx2str[idx.item()]
            predictions.append((char, prob.item()))
    
    return predictions

# Usage
# def get_khmer_text():
#     khmer_texts = []
#     with open('eng-khr_test.txt', 'r', encoding='utf-8') as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) > 1:
#                 khmer_texts.append(parts[1])
#     print(khmer_texts[:5])
#     print(len(khmer_texts))
#     return khmer_texts

# texts = get_khmer_text() # Your training texts

def get_khmer_text():
    """Get preprocessed Khmer text and split into train/val/test"""
    with open('data/processed_vocab.txt', 'r', encoding='utf-8') as f:
        text = f.read().strip()
        
    # Split into words and shuffle
    words = text.split()
    random.shuffle(words)
    
    # Get unique characters
    unique_chars = sorted(set(''.join(words)))
    print(f"Unique characters ({len(unique_chars)}):")
    for i, char in enumerate(unique_chars):
        print(f"{char}", end=' ')
        if (i + 1) % 10 == 0:  # New line every 10 chars
            print()
    print("\n")
    
    # Filter and create training data
    vocabs = [word for word in words if len(word) > 1]
    training_data = vocabs[:400000]
    
    # Split into train/val/test (80/10/10)
    train_data, temp_data = train_test_split(training_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Total examples: {len(training_data)}")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test_data)}")
    
    # Move to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'chars': unique_chars,
        'device': device
    }
    
if __name__ == "__main__":
    # Training code
    data = get_khmer_text()
    print("\nSample words from each split:")
    print(f"Train: {data['train'][:3]}")
    print(f"Val: {data['val'][:3]}")
    print(f"Test: {data['test'][:3]}")
    print("=" * 50)

    tokenizer = KhmerSyllableTokenizer()
    tokenizer.fit(data['train'])

    print(f"\nVocabulary size: {tokenizer.vocab_size}")

    config = NGramConfig() 
    config.epochs = 300 #200
    config.learning_rate = 0.05
    config.initial_temp = 1#0.8
    config.temp_decay = 0.9
    config.alpha = 0.3

    best_matrix, best_bias, history = build_ngram_model(tokenizer, data['train'], config)

    # Optional: Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss History')
    plt.legend()

    plt.subplot(132)
    plt.plot(history['lr'])
    plt.title('Learning Rate')

    plt.subplot(133)
    plt.plot(history['sparsity'])
    plt.title('Sparsity %')
    plt.tight_layout()
    plt.show()

    print("\nTransition Matrix Statistics:")
    print(f"Shape: {best_matrix.shape}")
    print(f"Non-zero transitions: {(best_matrix > 0).sum().item()}")
    print(f"Max probability: {best_matrix.max().item():.3f}")

    def test_prediction(text, tokenizer, trans_matrix, bias, k=5):
        # Remove last character for prediction
        input_text = text[:-1]
        actual_char = text[-1]
        
        predictions = predict_next_syllables(input_text, tokenizer, trans_matrix, best_bias)
        
        print("\nTest Results:")
        print(f"Input text: {input_text}")
        print(f"Actual next char: {actual_char}")
        print("\nModel predictions:")
        for char, prob in predictions:
            print(f"{char} -> {prob:.3f}" + (" ✓" if char == actual_char else ""))
        
        # Check if actual char is in top predictions
        pred_chars = [p[0] for p in predictions]
        if actual_char in pred_chars:
            rank = pred_chars.index(actual_char) + 1
            print(f"\nActual character found at rank {rank}")
        else:
            print("\nActual character not in top predictions")

    # Test examples
    test_examples = [
        "កម្ពុជា",
        "សួស្តី",
        "ជំរាបសួរ",
        "ស្វាគមន៍"
    ]
    print("\nTesting predictions on examples:")
    print("=" * 50)
    for example in test_examples:
        test_prediction(example, tokenizer, best_matrix, best_bias)
        print("-" * 50)

