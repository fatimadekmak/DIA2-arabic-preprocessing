import time
from clean_dataset import clean_dataset
import global_variables as gv
import os


if __name__ == "__main__" :
    start_time = time.time()
    gv.init()

    # create needed data logging folders
    if not os.path.exists("test\\stats"):
        os.makedirs("test\\stats")
    if not os.path.exists("test\\stats\\many_sentences_removed"):
        os.makedirs("test\\stats\\many_sentences_removed")
    if not os.path.exists("test\\stats\\short_docs"):
        os.makedirs("test\\stats\\short_docs")

    # clean dataset
    dataset_path = "test\data"
    dest_path = "test\data_filtered"
    clean_dataset(dataset_path,dest_path)
    

    print("****************************************************************************")
    print("****************************************************************************")
    print(f"Cleaning time = {time.time() - start_time}")