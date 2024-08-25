# utilities.py

import re
import itertools
import string
import arabic_reshaper
from bidi.algorithm import get_display
from langdetect import detect


def has_repeated_characters(text, threshold=0.25):
    """
    Detect if the text has repeated characters above a certain threshold.
    
    Args:
        text (str): The input text to analyze.
        threshold (float): The maximum allowed ratio of repeated characters.
        
    Returns:
        bool: True if the text has repeated characters above the threshold, False otherwise.
    """
    if len(text) == 0:
        return False
    # Count the occurrences of each character by grouping consecutive identical characters.
    char_counts = {char: len(list(group)) for char, group in itertools.groupby(text)}
    max_repeats = max(char_counts.values())
    return max_repeats / len(text) > threshold


def has_arabic_word_like_structures(text, min_word_like_sequences=5):
    """
    Check if the text contains a minimum number of Arabic word-like structures.
    
    Args:
        text (str): The input text to analyze.
        min_word_like_sequences (int): The minimum number of Arabic word-like sequences required.
        
    Returns:
        bool: True if the text contains the required number of Arabic word-like sequences, False otherwise.
    """
    # Find sequences of at least 2 Arabic characters.
    tokens = re.findall(r'[\u0600-\u06FF]{2,}', text)
    return len(tokens) >= min_word_like_sequences


# This helps in determining if the text is likely to be meaningful Arabic.
def contains_common_arabic_words(text, min_common_words=3):
    """
    Check if the text contains a minimum number of common Arabic words.
    
    Args:
        text (str): The input text to analyze.
        min_common_words (int): The minimum number of common Arabic words required.
        
    Returns:
        bool: True if the text contains the required number of common Arabic words, False otherwise.
    """
    # A set of commonly used Arabic words.
    common_arabic_words = {'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'أن', 'لا', 'ما', 'هو', 'كان', 'لم', 'يا', 
                           'قال', 'كل', 'هذه', 'وقد', 'كانت', 'لكن', 'وقال', 'بين', 'ذلك', 'يوم', 'منذ'}
    # Extract all Arabic words from the text.
    tokens = set(re.findall(r'[\u0600-\u06FF]+', text))
    # Count how many common Arabic words are present in the text.
    common_words_count = len(common_arabic_words.intersection(tokens))
    return common_words_count >= min_common_words


def is_primarily_arabic(text, threshold=0.6):
    """
    Determine if the text is primarily Arabic based on the proportion of Arabic characters.
    
    Args:
        text (str): The input text to analyze.
        threshold (float): The minimum proportion of Arabic characters required to consider the text as primarily Arabic.
        
    Returns:
        bool: True if the proportion of Arabic characters exceeds the threshold, False otherwise.
    """
    try:
        # Reshape the Arabic text for correct display.
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        
        # Count the number of Arabic characters.
        arabic_char_count = sum(1 for char in bidi_text if '\u0600' <= char <= '\u06FF')
        # Calculate the proportion of Arabic characters in the text.
        return arabic_char_count / len(bidi_text) > threshold
    except:
        return False


def is_detected_as_arabic(text):
    """
    Detect if the text is identified as Arabic by the langdetect library.
    
    Args:
        text (str): The input text to analyze.
        
    Returns:
        bool: True if the text is detected as Arabic, False otherwise.
    """
    try:
        return detect(text) == 'ar'
    except:
        return False


def has_excessive_punctuation(text, threshold=0.15):
    """
    Detect if the text contains excessive punctuation.
    
    Args:
        text (str): The input text to analyze.
        threshold (float): The maximum allowed ratio of punctuation characters.
        
    Returns:
        bool: True if the text contains punctuation above the threshold, False otherwise.
    """
    punct_count = sum(1 for char in text if char in string.punctuation)
    return punct_count / len(text) > threshold


def is_gibberish(text):
    """
    Determine if the text is considered gibberish based on several heuristics.
    
    Heuristics include:
    - Repeated characters
    - Excessive punctuation
    - Lack of Arabic word-like structures
    - Lack of common Arabic words
    - Not primarily Arabic by character count or language detection
    
    Args:
        text (str): The input text to analyze.
        
    Returns:
        bool: True if the text is considered gibberish, False otherwise.
    """
    return (
        has_repeated_characters(text) or
        has_excessive_punctuation(text) or
        (not has_arabic_word_like_structures(text) and not contains_common_arabic_words(text)) or
        (not is_primarily_arabic(text) and not is_detected_as_arabic(text))
    )
