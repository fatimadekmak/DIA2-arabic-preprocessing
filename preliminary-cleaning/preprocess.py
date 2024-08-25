from stats import print_stats, stats
from text_utils import *
import global_variables as gv


'''
preprocess_and_filter filters the given text using our defined preprocessing pipeline
returns the cleaned text or empty text if the document should be removed
stats and prints are for dataset statistics reporting purposes
'''
def preprocess_and_filter(text):
    stats["initial_nb_of_docs"] += 1
    sentences_per_document = 0 # number of sentences within this document
    sentences_removed = 0 # total number of sentences removed from this document

    text = remove_html_js(text)

    paragraphs = text.split('\n')  # Separate text into paragraphs
    paragraphs = [parag for parag in paragraphs if parag!=""]
    processed_paragraphs = []
    
    # split text into paragraphs
    for paragraph in paragraphs:
        # replace one or more whitespace by only one
        paragraph = normalize_text(paragraph) 
        paragraph = normalize_arabic(paragraph) 
        # split paragraph into sentences, preserving the punctuation marks
        sentences = separate_into_sentences(paragraph) # ? ! ...

        sentences_per_document += len(sentences)
        stats['initial_nb_of_sentences'] += len(sentences)
        
        # replacing PI by placeholders
        len_sentences = len(sentences) # number of sentences within this paragraph
        sentences = [remove_personal_info(sentence) for sentence in sentences]

        # removing excessive punctuation marks
        sentences = [successive_punctuation_check(sentence) for sentence in sentences]
        
        # removing long non-arabic spans
        sentences = [remove_long_non_arabic_spans(sentence) for sentence in sentences]
        
        # removing sentences with less than 70% arabic
        sentences = [sentence for sentence in sentences if is_arabic_sentence(sentence)]
        stats['sentences_removed_due_to_arabic_content'] += (len_sentences - len(sentences))
        sentences_removed += (len_sentences - len(sentences))
        len_sentences = len(sentences)

        # removing short sentences
        sentences = [sentence for sentence in sentences if word_count_check(sentence)]
        stats['sentences_removed_due_to_word_count'] += (len_sentences - len(sentences))
        sentences_removed += (len_sentences - len(sentences))
        len_sentences = len(sentences)

        stats['final_nb_of_sentences'] += len_sentences

        # non empty paragraph is stored
        if sentences:  # Only add non-empty paragraphs
            processed_paragraphs.append(' '.join(sentences))
        
    text = '\n'.join(processed_paragraphs)  # Rejoin paragraphs with newlines

    # document is removed if more that 30% of it's sentences were removed
    if sentences_per_document>0 and (sentences_removed/sentences_per_document)>0.3:
        print_stats(f"many_sentences_removed\\{gv.current_filename}",f"{sentences_removed*100/sentences_per_document}% of its sentences were removed\n{text}\n","w")
        text = ''
        stats['documents_removed_due_to_sentence_removal'] += 1

    # document is removed if it's too short 
    elif len(text.replace('\n', ' ').split(' ')) < 64:
        print_stats(f"short_docs\\{gv.current_filename}",f"{text}\n","w")
        text = ''
        stats['documents_removed_due_to_length'] += 1
    
    return text