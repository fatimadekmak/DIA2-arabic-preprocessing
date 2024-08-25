stats = {
    'initial_nb_of_sentences': 0,
    'final_nb_of_sentences':0,
    'initial_nb_of_docs': 0,
    'removed_personal_info': 0,  # Track the number of occurrences removed, not words
    'sentences_removed_due_to_arabic_content': 0,
    'sentences_removed_due_to_word_count': 0,
    'sentences_modified_due_to_successive_punctuation': 0,
    'sentences_removed_due_to_long_non_arabic_span': 0,
    'documents_cleaned_from_html': 0,
    'documents_cleaned_from_js': 0,
    'documents_removed_due_to_length': 0,
    'documents_removed_due_to_sentence_removal': 0,
}

def print_stats(filename, text, options="w"):
    with open(f"test\\stats\\{filename}", options, encoding='utf-8') as f:
        f.write(text);
        f.write("\n")