import re
from pathlib import Path

def clean_khmer_text(text):
    """Keep only Khmer Unicode characters"""
    # Remove Khmer full stop (។)
    text = text.replace('។', '')
    
    # Keep only Khmer Unicode range (1780-17FF)
    cleaned_chars = []
    for char in text:
        if '\u1780' <= char <= '\u17FF':
            cleaned_chars.append(char)
        elif char == ' ':  # Keep spaces between words
            cleaned_chars.append(char)
    
    return ''.join(cleaned_chars)

def process_file(input_file, output_file):
    """Process input file and create clean space-separated vocabulary"""
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean the text
    cleaned_text = clean_khmer_text(text)
    
    # Remove multiple spaces and trim
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Save processed text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    # Print sample
    print(f"Sample of cleaned text (first 100 chars):")
    print(cleaned_text[:100])

if __name__ == "__main__":
    input_file = "khmer_spm_pretokenized.txt"
    output_file = "data/processed_vocab.txt"
    
    Path("data").mkdir(exist_ok=True)
    process_file(input_file, output_file)