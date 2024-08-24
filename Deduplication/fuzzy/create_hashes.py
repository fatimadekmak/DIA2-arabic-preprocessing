import os
import re
import time
import pandas as pd
from datasketch import MinHash
from nltk import word_tokenize
from pandarallel import pandarallel
import pickle

# Initialize pandarallel for parallel processing with a progress bar
pandarallel.initialize(progress_bar=True)

def preprocess_and_tokenize(text_array):
    """
    Preprocess the text by removing punctuation and diacritics, then tokenize it.
    
    Args:
        text_array (list): List of text strings to preprocess and tokenize.
        
    Returns:
        list: List of tokens after preprocessing.
    """
    all_tokens = []
    for text in text_array:
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove Arabic diacritics
        text = re.sub(r'[\u064B-\u065F]', '', text)
        # Tokenize the text
        tokens = word_tokenize(text)
        all_tokens.extend(tokens)
    return all_tokens

def process_text(row, num_perm):
    """
    Process a single text by creating a MinHash object from its tokens.
    
    Args:
        row (pandas.Series): A row from the DataFrame containing 'text' and 'unique_id'.
        num_perm (int): Number of permutations for the MinHash.
        
    Returns:
        tuple: The unique_id and the corresponding MinHash object.
    """
    text = row['text']
    unique_id = row['unique_id']
    tokens = preprocess_and_tokenize(text)
    m = MinHash(num_perm=num_perm)
    for token in tokens:
        m.update(token.encode('utf8'))
    return unique_id, m

def parallel_create_minhashes(df, num_perm = 32):
    """
    Use pandarallel to process the DataFrame and create MinHashes in parallel.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the text and unique_id columns.
        num_perm (int): Number of permutations for the MinHash.
        
    Returns:
        dict: A dictionary where keys are unique_ids and values are MinHash objects.
    """
    results = df.parallel_apply(process_text, axis=1, num_perm=num_perm)
    minhashes = {unique_id: minhash for unique_id, minhash in results}
    return minhashes

def main():
    """
    Main function to load data, process it, and save MinHashes.
    """
    # directory containing the Parquet files (modify as needed)
    data_path = "/path/to/Cleaned/Data/"
    
    # Loop through all files in the directory
    for file in os.listdir(data_path):
        
        print("Processing file:", file)
        
        df = pd.read_parquet(os.path.join(data_path, file))
        
        # a unique identifier for each text is required for deduplication
        if 'unique_id' not in df.columns:
            print(f"WARNING: 'unique_id' column missing in {file}. Skipping...")
            continue
        
        # path to save the MinHash objects of the texts included in the file
        hash_filename = f"/path/to/hashes/{file}.pkl"
        
        # Check if the hash file already exists
        if os.path.exists(hash_filename):
            print(f"File {hash_filename} already exists. Skipping...")
            continue
        
        # Process the DataFrame to create MinHashes
        start = time.time()
        minhashes = parallel_create_minhashes(df[['text', 'unique_id']])
        print("Time taken:", time.time() - start, "seconds for processing", len(df), "texts.")
        
        print("Dumping to hash file...")
        start = time.time()
        
        # Save the MinHashes to a pickle file
        with open(hash_filename, 'wb') as file:
            pickle.dump(minhashes, file)
        print("Time taken to dump into file:", time.time() - start, "seconds.")
        
        # Free up memory by deleting the MinHashes
        del minhashes

if __name__ == '__main__':
    main()
