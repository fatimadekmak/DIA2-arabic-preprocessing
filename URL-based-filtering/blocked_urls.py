import pandas as pd
import tldextract
import os

# Load Blacklists
# Source: https://dsi.ut-capitole.fr/blacklists/index_en.php
blacklists = {
    'adult': set(open('/path/to/blacklist/adult_domains').read().splitlines()),
    'agressif': set(open('/path/to/blacklist/agressif_domains').read().splitlines())
}

def get_blacklist_with_reason(url):
    """
    Determine the blacklist category of a given URL.
    
    Args:
        url (str): The URL to check against blacklists.
    
    Returns:
        str: The reason for blacklisting (e.g., 'adult', 'agressif') if found, otherwise None.
    """
    extracted = tldextract.extract(url)
    domain = "{}.{}".format(extracted.domain, extracted.suffix)
    for reason, blist in blacklists.items():
        if domain in blist:
            return reason
    return None

def process_parquet_file(file_path, output_dir, removed_urls_dir):
    """
    Process a single parquet file, removing blacklisted URLs and saving the cleaned data and removed URLs.
    
    Args:
        file_path (str): The path to the parquet file.
        output_dir (str): Directory to save the cleaned parquet file.
        removed_urls_dir (str): Directory to save the list of removed URLs.
    """
    print(f"Processing {file_path}")
    
    # Load the data from the Parquet file
    data = pd.read_parquet(file_path)
    
    # Filtering URLs and organizing by blacklist reason
    urls_to_remove = {}
    for index, row in data.iterrows():
        reason = get_blacklist_with_reason(row['url'])
        if reason:
            urls_to_remove.setdefault(reason, []).append(row['url'])

    print(f">>>> Number of rows in {os.path.basename(file_path)} is {len(data)}")
    
    # Removing blacklisted URLs from the dataset
    for urls in urls_to_remove.values():
        data = data[~data['url'].isin(urls)]

    print(f">>>> Number of rows in {os.path.basename(file_path)} after cleaning is {len(data)}")
    
    # Save the cleaned data to a new Parquet file
    cleaned_file_path = os.path.join(output_dir, os.path.basename(file_path))
    data.to_parquet(cleaned_file_path)
    
    # Save the list of removed URLs to a text file, categorized by reason
    removed_urls_file_path = os.path.join(removed_urls_dir, f"{os.path.basename(file_path)}.txt")
    with open(removed_urls_file_path, 'w') as f:
        for reason, urls in urls_to_remove.items():
            f.write(f"Reason for removal: {reason}\n" + "\n".join(urls) + "\n\n")

    # Print the total number of removed URLs
    total_removed = sum(len(urls) for urls in urls_to_remove.values())
    print(f"    >>>> Total URLs removed: {total_removed}")
    
    # Print the count of URLs removed for each reason
    for reason, urls in urls_to_remove.items():
        print(f"        >>>> {reason}: {len(urls)}")

def main():
    """
    Main function to load datasets, find duplicates, and process CulturaX files.
    """
    # Define paths (replace with actual paths)
    input_dir = "/path/to/CulturaX/"  # Directory containing the parquet files
    output_dir = "/path/to/domain_filtered_CulturaX/"  # Directory to save cleaned files
    removed_urls_dir = "/path/to/removed_urls/"  # Directory to save lists of removed URLs
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(removed_urls_dir, exist_ok=True)
    
    # Loop over parquet files in the input directory
    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            file_path = os.path.join(input_dir, file)
            process_parquet_file(file_path, output_dir, removed_urls_dir)

    print("Done!")

if __name__ == "__main__":
    main()
