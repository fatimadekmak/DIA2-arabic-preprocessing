import pandas as pd
import os

def load_parquet_files_from_folder(folder_path):
    """
    Load all parquet files from a given folder into a single DataFrame.
    
    Args:
        folder_path (str): Path to the folder containing parquet files.
        
    Returns:
        pandas.DataFrame: A DataFrame concatenated from all parquet files in the folder.
    """
    dataframes = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            df = pd.read_parquet(file_path)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def deduplicate_cultura_file(cultura_df, other_urls, url_column='url'):
    """
    Remove rows from the CulturaX DataFrame where the URL exists in other datasets.
    
    Args:
        cultura_df (pandas.DataFrame): The DataFrame to be deduplicated.
        other_urls (array-like): List or array of URLs to be removed from cultura_df.
        url_column (str): The column name containing URLs.
        
    Returns:
        tuple: A tuple containing the deduplicated DataFrame and the number of rows removed.
    """
    # Remove rows where the URL exists in the other datasets
    deduplicated_df = cultura_df[~cultura_df[url_column].isin(other_urls)]
    length_diff = len(cultura_df) - len(deduplicated_df)
    
    return deduplicated_df, length_diff

def process_cultura_files(cultura_folder, other_urls, output_suffix='_no_url_dups.parquet'):
    """
    Process each CulturaX parquet file one by one, removing duplicates based on URLs.
    
    Args:
        cultura_folder (str): Path to the folder containing CulturaX files.
        other_urls (array-like): List or array of URLs to be removed from CulturaX files.
        output_suffix (str): Suffix to be added to the output files.
    """
    total_removed = 0
    for root, dirs, files in os.walk(cultura_folder):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            
            # Load the CulturaX file
            cultura_df = pd.read_parquet(file_path)
            
            # Deduplicate based on URLs
            deduplicated_df, rem = deduplicate_cultura_file(cultura_df, other_urls)
            total_removed += rem
            
            # Save the deduplicated CulturaX file
            output_file = os.path.join(root, file.replace('.parquet', output_suffix))
            deduplicated_df.to_parquet(output_file, index=False)
            print(f"Saved deduplicated file: {output_file} with {rem} duplicates removed.")
    
    print(f"Total number of documents removed due to duplication: {total_removed}")

def main(cultura_folder, other_datasets_folders):
    """
    Main function to load datasets, find duplicates, and process culturaX Data files.
    
    Args:
        cultura_folder (str): Path to the folder containing CulturaX files.
        other_datasets_folders (list of str): List of paths to folders containing other datasets.
    """
    # Load and concatenate all other datasets
    other_datasets = [load_parquet_files_from_folder(folder) for folder in other_datasets_folders]
    combined_df = pd.concat(other_datasets, ignore_index=True)
    
    # Extract the list of URLs from the other datasets
    other_urls = combined_df['url'].unique()
    
    # Process CulturaX files
    process_cultura_files(cultura_folder, other_urls)
    

# Example usage:
if __name__ == "__main__":
    # duplicate urls to be removed from CulturaX files
    cultura_folder = "/path/to/Data1"  # Replace with actual path
    other_datasets_folders = [
        "/path/to/Final/Data2",  # Replace with actual paths
        "/path/to/Final/Data3",
        "/path/to/Final/Data4"
    ]
    
    main(cultura_folder, other_datasets_folders)
