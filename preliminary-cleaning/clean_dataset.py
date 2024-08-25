import os
from preprocess import preprocess_and_filter
from stats import stats
import json
import global_variables as gv
import pandas as pd


def clean_dataset(dataset_path,dest_path):
    for filename in os.listdir(dataset_path):

        if filename.endswith('.txt') == False:
            continue
        print("processing file ",filename)
        file_path = os.path.join(dataset_path, filename)
        gv.current_file = file_path
        gv.current_filename = filename

        #load dataset json
        with open(file_path, 'r', encoding="utf-8") as file:
            text = file.read()

        text = preprocess_and_filter(text)
        if(text==""):
            print(f"{filename} was deleted")
            continue

        # Save the filtered data in another json file
        with open(os.path.join(dest_path, f"{filename}"), 'w', encoding="utf-8") as file:
            file.write(text)
        print(f"Saved filtered data to new {filename}")
        
    # Open the file in write mode and write the text
    with open('test\\stats\\statistics.json', 'w') as file:
        json.dump(stats, file, indent=4)