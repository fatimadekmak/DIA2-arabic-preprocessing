# DIA2 Arabic Preprocessing

This repository contains the code used for preprocessing and creating the DIA2 dataset—a diverse diacritized Arabic dataset suitable for training Large Language Models (LLMs) and other Natural Language Processing (NLP) tasks. The repository is organized into several subfolders, each corresponding to a specific preprocessing step, as illustrated in the flowchart provided. ![Preprocessing Flowchart](/Arabic%20Text%20Preprocessing%20Flowchart.jpeg)

## Table of Contents
- [Repository Structure](#repository-structure)
  - [Preliminary Cleaning](#1-preliminary-cleaning)
  - [URL-based Filtering](#2-url-based-filtering)
  - [Metric-based Filtering](#3-metric-based-filtering)
  - [Arabic Processing and Diacritization](#4-arabic-processing-and-diacritization)
  - [Deduplication](#5-deduplication)
- [Citing the Repository](#citing-the-repository)

## Repository Structure

The repository is structured into the following main subfolders:

### 1. `preliminary-cleaning/`
This subfolder contains scripts that handle the initial data cleaning steps. These scripts remove unnecessary and problematic content from the dataset, ensuring that only relevant and clean data proceeds to the next stages. The steps included are:

- **HTML and JavaScript Removal**: Strips out HTML tags and JavaScript code to ensure only plain text remains. 
- **Personal Information Removal**: Replaces any personally identifiable information from the dataset by a placeholder. 
```
input:
ينصح المستخدمين بإعادة توجيه رسائل البريد الإلكتروني لمشبوهة إلى نطاق spoof@PayPal.com

output:
ينصح المستخدمين بإعادة توجيه رسائل البريد الإلكتروني لمشبوهة إلى نطاق Example@mail.com

```
- **Excessive Punctuation Marks Removal**: Reduces noise by removing excessive punctuation.
```
input:
لقد أزحتها إلى الخارج بعصاي!!!!!!

output:
لقد أزحتها إلى الخارج بعصاي!!!

```
- **Removal of Long Non-Arabic Spans**: Filters out non-Arabic text spans that exceed a certain length.
```
input:
الموقع الإلكتروني لعلم اللغويات البيئية (www-gewi.com)

output:
الموقع الإلكتروني لعلم اللغويات البيئية 

```
- **Sentence Filtering**: Removes sentences with less than 70% Arabic content, short sentences (less than 7 words), and corrupted documents where more than 30% of sentences were removed.
- **Short Document Removal**: Discards documents that are too short (less than 64 words).

**Prerequisites**: 
- Python
- Libraries: `BeautifulSoup`, `phonenumbers`, `regex`
- Data Format: The code works with data in plain text format. It can be modified as needed.

### 2. `URL-based-filtering/`
This folder contains scripts used to filter out content based on URLs. This step ensures that any content from irrelevant or untrusted sources is excluded from the dataset. The [UT1 blacklists](https://dsi.ut-capitole.fr/blacklists/index_en.php) were utilized in this process to identify and filter out URLs from known sources of adult and agressive content.

**Prerequisites**:
- Python
- Libraries: `tldextract`
- Data Format: The code works with data in parquet format containing a 'url' column. It can be modified as needed.

### 3. `metric-based-filtering/`
Scripts in this subfolder perform filtering based on various metrics such as document length, sentence quality, and Arabic language content. This step is crucial for ensuring the quality of the dataset by removing low-quality and irrelevant content.

**Prerequisites**:
- Python
- Libraries: langdetect from PyPi
- Data Format: The code works with data in parquet format containing the text in a 'text' column. It can be modified as needed.

### 4. `arabic-processing-and-diacritization/`
This subfolder handles the final Arabic text processing and diacritization steps. It includes:

- **Non-Arabic Characters Removal**: Ensures that the text contains only Arabic characters.
- **Repeated Characters Handling**: Limits repeated Arabic characters to two to standardize the text.
```
أسسست to أسست
الللبنانية to اللبنانية
مبرووووك to مبرووك
```
- **Punctuation Normalization**: Replaces English punctuation marks with Arabic ones.
```
? to ؟
, to ،
; to ؛
```
- **Unicode Normalization**: Standardizes the text to a consistent Unicode format.
- **Diacritization**: Adds diacritics to the Arabic text using the [CATT (Character-based Arabic Tashkeel Transformer)](https://github.com/abjadai/catt) encoder-only model, selected after a comparative evaluation against [the CAMeL Tools' Maximum Likelihood Estimation (MLE) Disambiguator.](https://camel-tools.readthedocs.io/en/latest/api/disambig/mle.html) (see paper for details).
- **Pre-diacritization normalization**: Due to model incompatibility, and before diacritization, Persian-origin characters are normalized to MSA equivalents and numeric tokens are expanded to Arabic words using `num2words`. Original digits are then restored post-diacritization via index mapping. Long sequences are chunked at punctuation boundaries to respect the model's 1,024-character limit.
```
Input:
وأشار وكيل وزارة التربية والتعليم، إلى أن الاختبارات ستمتد اعتبارًا من يوم الأحد 

Output:
وَأَشارَ وَكِيلُ وِزارَةِ التَرْبِيَةِ وَالتَعْلِيمِ، إِلَى أَنَّ الاِخْتِباراتِ سَتَمْتَدّ اِعْتِباراً مِن يَوْمِ الأَحَد

```

**Prerequisites**:
- Python
- Libraries: `camel_tools`, `pandarallel`:The code is parallelized using `pandarallel` to take advantage of multiple CPU cores, speeding up the text processing significantly.
- Data Format: The code works with data in parquet format containing the text in a 'text' column. It can be modified as needed.

### 5. `deduplication/`
The purpose of these scripts is to efficiently detect duplicate texts from large datasets. The process includes creating MinHashes for each document, building LSH indexes, identifying near duplicates, in addition to deduplicating datasets based on duplicate URLs. These techniques are particularly useful when working with massive datasets where traditional deduplication methods would be computationally expensive.

**Prerequisites**:
- Python
- Libraries: `pandas`, `datasketch`
- Data Format: The code works with data in parquet format containing the text in a 'text' column, unique identifier for each text in a 'unique_id' column, and the url in a 'url' column. It can be modified as needed.

### create_hashes.py

**Purpose**:

MinHashes are a compact representation of the content of a document that can be used to efficiently compare documents.This script generates MinHash signatures for each document in your dataset. It saves the minhashes of the documents belonging to one parquet in one pkl file.

**Usage**:

Run this script on your dataset to create MinHash signatures for each document. The generated MinHashes are stored in pickle files for further processing. Make sure each document in your dataset has a unique identifier which is necessary for deduplication.

### create_lsh.py

**Purpose**:

This script takes the MinHash signatures generated by `create_hashes.py` and builds an LSH (Locality-Sensitive Hashing) index. LSH allows for efficient nearest neighbor search, which is used to identify potential duplicates in large datasets. 

**Usage**:

Run this script after generating MinHash signatures. It will create an LSH index that can be used to find duplicate documents. You can either create one LSH index for the whole dataset, or multiple LSH indexes for each shard if you divided your dataset into shards prior to deduplication.

### get_dups.py

**Purpose**:

This script uses the LSH index created by `create_lsh.py` to identify duplicates within and across datasets. It compares the MinHashes stored in the LSH index and identifies pairs of documents that are likely to be duplicates.

**Usage**:

Run this script to identify duplicate documents based on their MinHash signatures. The script will output a list of duplicate pairs.

### url_dedup.py

**Purpose**:

This script deduplicates datasets based on URLs. It removes records with duplicate URLs, ensuring that only unique records are retained in the final dataset. In the provided code, duplicate urls are detected from selected datasets and removed from CulturaX dataset.

**Usage**:

Run this script to remove records with duplicate URLs from your dataset.

## Citing the Repository

If you use this code in your research, please cite the following paper:
```bibtex
@inproceedings{dekmak2026dia2,
  title     = {DIA2: A Comprehensive and Diverse Diacritized Modern Standard Arabic Corpus for Large-Scale NLP Research},
  author    = {Dekmak, Fatima and Elbassuoni, Shady and Shaban, Khaled and Hajj, Hazem and El Hajj, Wassim and Abu Adla, Yasmine and Alabrash, Buthaina},
  booktitle = {Proceedings of the OSACT7 Workshop at LREC-COLING 2026},
  year      = {2026}
}
```
