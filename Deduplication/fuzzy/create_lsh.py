import os
import pickle
from datasketch import MinHashLSH
from tqdm import tqdm
import gc

def load_minhash(file_path):
    """
    Load a MinHash object from a pickle file.
    
    Args:
        file_path (str): Path to the pickle file containing MinHash objects.
        
    Returns:
        dict: A dictionary with keys and their corresponding MinHash objects.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)

def create_lsh_for_shard(shard_files, threshold=0.5, num_perm=32):
    """
    Create an LSH index for a given set of MinHash objects from shard files.
    
    Args:
        shard_files (list of str): List of file paths containing MinHash objects.
        threshold (float, optional): The Jaccard similarity threshold. Default is 0.5.
        num_perm (int, optional): Number of permutations for the MinHash. Default is 32.
        
    Returns:
        MinHashLSH: An LSH index populated with the MinHash objects from the shard files.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    for file_path in tqdm(shard_files, desc="Processing files in shard"):
        minhashes = load_minhash(file_path)
        
        for key, minhash in tqdm(minhashes.items(), total=len(minhashes), desc="Inserting minhashes"):
            lsh.insert(key, minhash)
        
        del minhashes
        gc.collect()  # Explicit garbage collection to free up memory
    
    return lsh

def main(hashes_directory, output_directory):
    """
    Main function to process a shard and create an LSH index.
    
    Args:
        hashes_directory (str): Directory containing shard files with MinHash objects.
        output_directory (str): Directory where the LSH index will be saved.
    """
    shard_id = hashes_directory.split('_')[-1]  # Extract the shard ID from the directory name
    output_path = os.path.join(output_directory, f"lsh_shard_{shard_id}.pkl")

    # Skip if the LSH index already exists for this shard
    if os.path.exists(output_path):
        print(f"LSH index for shard {shard_id} already exists, skipping.")
        return

    shard_files = [os.path.join(hashes_directory, file) for file in os.listdir(hashes_directory) if file.endswith(".pkl")]

    print(f"Processing shard {shard_id} with {len(shard_files)} files")

    lsh = create_lsh_for_shard(shard_files)

    # Save the LSH index to the output path
    with open(output_path, "wb") as f:
        pickle.dump(lsh, f)
    
    print(f"LSH index for shard {shard_id} saved to {output_path}")

if __name__ == "__main__":
    # create LSH for a collection of minhashes
    shard_minhashes_path = f"/path/to/hashes/shard_shard_id"  # Update this path as needed
    output_directory = "/path/to/lsh_dir"  # Update this path as needed
    main(shard_minhashes_path, output_directory)
