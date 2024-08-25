import json
import time
import os
import pandas as pd
import re
from camel_tools.utils import normalize
from camel_tools.utils.charsets import AR_CHARSET, AR_LETTERS_CHARSET
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
from pandarallel import pandarallel

# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

# Define the set of Arabic characters to keep
arabic_letters_diacritics = ''.join(AR_CHARSET)
allowed_chars = (
    arabic_letters_diacritics +
    '٠١٢٣٤٥٦٧٨٩'  # Arabic numbers
    '0123456789'  # English numbers
    '؟؛،.!?,;$%'  # Punctuation marks
    ' \n'
)

# Regex pattern to match any character that is not in the allowed set
non_arabic_pattern = re.compile(f'[^{allowed_chars}]')

def clean_arabic_text(text):
    """
    Clean the Arabic text by removing unwanted characters and extra spaces.
    """
    cleaned_text = non_arabic_pattern.sub(' ', text)
    cleaned_text = re.sub(r' \t+', ' ', cleaned_text)
    return cleaned_text

# Define the set of Arabic letters
arabic_letters = ''.join(AR_LETTERS_CHARSET).replace("ـ", "")

def normalize_arabic_text(text):
    """
    Normalize Arabic text using CAMeL Tools, handling elongations and replacing punctuation.
    """
    normalized_text = normalize.normalize_unicode(text)
    
    # Handle elongations (e.g., مبروووووووك to مبروك)
    elongation_pattern = re.compile(r'([' + re.escape(arabic_letters) + r'])\1{2,}')
    normalized_text = elongation_pattern.sub(r'\1\1', normalized_text)
    
    # Remove Kasheeda (e.g., العـــــربية to العربية)
    normalized_text = re.sub(r'\u0640+', '', normalized_text)
    
    # Replace English punctuation with Arabic punctuation
    normalized_text = normalized_text.replace('?', '؟').replace(';', '؛').replace(',', '،')
    
    return normalized_text

# Initialize the MLE Disambiguator
mle = MLEDisambiguator.pretrained()

def add_diacritics(text):
    """
    Add diacritics to Arabic text using CAMeL Tools' MLE Disambiguator.
    """
    try:
        global mle
        lines = text.split('\n')
        diacritized_lines = []
        for line in lines:
            tokenized_text = simple_word_tokenize(line)
            disambiguated_text = mle.disambiguate(tokenized_text)
            diacritized_line = ' '.join(
                [d.analyses[0].analysis['diac'] if d.analyses else '' for d in disambiguated_text]
            )
            diacritized_lines.append(diacritized_line)
        return '\n'.join(diacritized_lines)
    except Exception as e:
        print(f"Error processing text: {e}")
        return text

def preprocess_text(text):
    """
    Preprocess the Arabic text by cleaning, normalizing, and adding diacritics.
    """
    text = clean_arabic_text(text)
    text = normalize_arabic_text(text)
    return add_diacritics(text)

def process_parquet_file(input_filepath):
    """
    Process a single parquet file, applying text preprocessing to each row.
    """
    df = pd.read_parquet(input_filepath)
    print(f"Started preprocessing {input_filepath}")
    start = time.time()
    
    # Apply preprocessing in parallel
    df['text'] = df['text'].parallel_apply(preprocess_text)
    
    print(f"Time taken: {time.time() - start} seconds for processing {len(df)} texts.")
    
    # Save the processed DataFrame
    output_filepath = input_filepath.replace(".parquet", "_processed.parquet")

    
    df.to_parquet(output_filepath, index=False)
    print(f"Saved processed file to {output_filepath}")

def main():
    """
    Main function to iterate over all parquet files in a directory and process them.
    """
    input_folder = r"/path/to/your/folder"  # Update this path to the folder containing parquet files
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".parquet"):
            input_filepath = os.path.join(input_folder, filename)
            process_parquet_file(input_filepath)

if __name__ == '__main__':
    main()
