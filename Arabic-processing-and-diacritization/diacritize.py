import glob
import os
import re
import time
import pandas as pd
from my_utils import (
    load_model,
    normalize_non_core_arabic,
    remove_non_arabic_modified,
    numeric_expansion_and_record,
    symbol_expansion_and_record,
    split_by_punc_keep,
    split_long_segment,
    build_index_mapping,
    restore_symbols_after_diacritization,
    restore_digits_after_diacritization,
)

# Configuration

INPUT_FOLDER  = "/path/to/input/parquet/folder"               # folder containing input parquet files
OUTPUT_FOLDER = "/path/to/input/parquet/folder_diacritized"   # folder to save diacritized output
MODEL_PATH    = "models/best_eo_mlm_ns_epoch_193.pt"
MAX_SEQ_LEN   = 1024
BATCH_SIZE    = 128
SAVE_INTERVAL = 3600  # save checkpoint every hour


# Core diacritization function

def diacritize_text(raw_text: str, model, max_seq_len: int) -> str:
    """
    Diacritize a single Arabic text string using the CATT encoder-only model.

    The function handles:
    - Unicode normalization and non-Arabic character removal
    - Numeric and symbol span tracking for post-diacritization restoration
    - Chunking long sequences to respect the model's maximum sequence length
    - Batched inference for efficiency
    """
    raw_text = raw_text.strip(" ")
    raw_text = normalize_non_core_arabic(raw_text)
    raw_text, _ = remove_non_arabic_modified(raw_text)
    raw_text = re.sub(r' {2,}', ' ', raw_text)

    # Record and expand numeric and symbol spans before diacritization
    X0, numeric_spans = numeric_expansion_and_record(raw_text)
    X1, symbol_spans  = symbol_expansion_and_record(X0)

    # Split on punctuation, keeping punctuation tokens separate
    pieces = split_by_punc_keep(X1)

    # Build chunks that respect max_seq_len
    chunks = []
    for idx, piece in enumerate(pieces):
        if idx % 2 == 1:
            # Punctuation token: keep as-is
            chunks.append(piece)
        else:
            if len(piece) <= max_seq_len:
                chunks.append(piece)
            else:
                sub_segs = split_long_segment(piece, max_seq_len)
                for j, sub in enumerate(sub_segs):
                    chunks.append(sub)
                    if j < len(sub_segs) - 1:
                        chunks.append(" ")

    # Batch-diacritize all real text chunks (even-indexed)
    real_idxs   = [i for i in range(len(chunks)) if i % 2 == 0]
    real_chunks = [chunks[i] for i in real_idxs]
    diac_real   = model.do_tashkeel_batch(real_chunks, batch_size=BATCH_SIZE, verbose=False)
    real_iter   = iter(diac_real)

    diacritized = []
    for idx, chunk in enumerate(chunks):
        if idx % 2 == 1:
            diacritized.append(chunk)
        else:
            diacritized.append(next(real_iter))

    # Reassemble and restore original numeric and symbol spans
    X2         = "".join(diacritized)
    mapping    = build_index_mapping(X1, X1, X2)
    X2         = restore_symbols_after_diacritization(X2, symbol_spans, mapping)
    mapping    = build_index_mapping(X0, X0, X2)
    final_text = restore_digits_after_diacritization(X2, numeric_spans, mapping)

    return final_text


# File processing

def process_parquet_file(in_path: str, out_path: str, model) -> None:
    """
    Diacritize all rows in a parquet file and save the result.
    Checkpoints are saved periodically to avoid data loss on long runs.
    """
    print(f"Processing: {os.path.basename(in_path)}")
    df = pd.read_parquet(in_path)
    df["diacritized_text"] = ""
    start_time = time.time()

    try:
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Row {idx}/{len(df)} | Elapsed: {elapsed:.0f}s")

            df.at[idx, "diacritized_text"] = diacritize_text(
                row["text"], model, MAX_SEQ_LEN
            )

            # Periodic checkpoint save
            if time.time() - start_time > SAVE_INTERVAL:
                df.to_parquet(out_path, index=False)
                print(f"  Checkpoint saved at row {idx} after {time.time() - start_time:.0f}s")

    except Exception as e:
        import traceback
        print(f"Error at row {idx}: {e}")
        traceback.print_exc()

    finally:
        df = df.drop(columns=["text"])
        df.to_parquet(out_path, index=False)
        print(f"Saved: {out_path} ({len(df)} rows, {time.time() - start_time:.0f}s)")


# Main

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    model = load_model(MODEL_PATH, num_layers=6, max_seq_len=MAX_SEQ_LEN)

    parquet_files = glob.glob(os.path.join(INPUT_FOLDER, "*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in: {INPUT_FOLDER}")
        return

    for in_path in parquet_files:
        base     = os.path.basename(in_path)
        out_path = os.path.join(OUTPUT_FOLDER, base.replace(".parquet", "_diacritized.parquet"))

        if os.path.exists(out_path):
            print(f"Skipping (already processed): {base}")
            continue

        process_parquet_file(in_path, out_path, model)

    print("All files processed.")


if __name__ == "__main__":
    main()