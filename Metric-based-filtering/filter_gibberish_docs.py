import os
import time
import pandas as pd
import json
from utilities import is_gibberish

# Function to process the text data in the DataFrame and identify gibberish texts.
def process_texts(df):
    """
    Process a DataFrame of texts to identify gibberish texts.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing a 'text' column.
        
    Returns:
        list: A list of tuples containing the index and a boolean indicating if the text is gibberish.
        list: A list of tuples containing the index and the gibberish text.
    """
    results = []
    gibberish_examples = []
    for index, row in df.iterrows():
        text = row['text']
        is_gibberish_text = is_gibberish(text)
        results.append((index, is_gibberish_text))
        if is_gibberish_text:
            gibberish_examples.append((index, text))
    return results, gibberish_examples

# Function to save gibberish texts to files for review or further analysis.
def save_gibberish_texts(gibberish_examples, folder, parquet_file):
    """
    Save sample gibberish texts to individual files for further analysis.
    
    Args:
        gibberish_examples (list): A list of tuples containing the index and the gibberish text.
        folder (str): The folder where the original parquet file is located.
        parquet_file (str): The name of the original parquet file (without extension).
    """
    output_dir = 'gibberish_texts'
    os.makedirs(output_dir, exist_ok=True)
    for index, text in gibberish_examples:
        file_name = f"{folder}_{parquet_file}_{index}.txt"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)

# Main function to process all datasets, clean gibberish texts, and save the cleaned datasets.
def main():
    """
    Main function to iterate through all dataset folders, identify and remove gibberish texts,
    and save the cleaned datasets along with logging gibberish examples.
    """
    dataset_path = '/path/to/Data/'  # Update this path as needed
    stats = {
        "total_number_gibberish": 0,  # Counter for total gibberish texts identified
        "total_number_total": 0       # Counter for total texts processed
    }
    start = time.time()

    for folder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, folder)
        print("> processing folder " + folder)
        
        file_start = time.time()
        # Loop through each file in the subfolder
        for file in os.listdir(subfolder_path):
            if not file.endswith(".parquet"):
                continue
                
            cleaned_file = file.replace(".parquet", "_cleaned.parquet")
            cleaned_filepath = os.path.join(subfolder_path, cleaned_file)
            
            # Skip processing if the cleaned file already exists
            if os.path.exists(cleaned_filepath):
                print(f">> skipping {file} as cleaned file {cleaned_file} already exists")
                continue
            
            print(f">> processing {file}")
            filepath = os.path.join(subfolder_path, file)
            df = pd.read_parquet(filepath)
            
            # Process texts to determine gibberish
            results, gibberish_examples = process_texts(df)
    
            new_df = []
            # Remove gibberish from df
            for index, gibberish in results:
                if not gibberish:
                    new_df.append(df.iloc[index])
                else:
                    stats["total_number_gibberish"] += 1
                    
            df_cleaned = pd.DataFrame(new_df)
            stats['total_number_total'] += df_cleaned.shape[0]

            # Save cleaned DataFrame and gibberish
            df_cleaned.to_parquet(cleaned_filepath, index=False)
            # Log some gibberish examples
            save_gibberish_texts(gibberish_examples, folder, file.split(".parquet")[0])
            del df, df_cleaned

        print("Time taken:", time.time() - file_start, "seconds for processing", folder)
    
    # Save statistics to a JSON file
    with open('stats.json', 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"All files processed in {time.time() - start} seconds")

if __name__ == '__main__':
    main()
