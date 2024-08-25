import re
import unicodedata
from bs4 import BeautifulSoup
import phonenumbers
from stats import print_stats,stats
import global_variables as gv

# remove sentences with less that 70% arabic content
def is_arabic_sentence(sentence, threshold=0.7):
    if(sentence==''):
        return False
    allowed_chars = '0123456789,.\'\"()؛?!؟:;`،'
    arabic_count = sum(1 for char in sentence if '\u0600'<char<'\u06FF' or char in allowed_chars) # count arabic chars + add more neutral characters
    non_space_chars = [char for char in sentence if char.strip()] # count nb of chars
    if(arabic_count / max(len(non_space_chars),1) < threshold) :
        print_stats("sentences_with_low_arabic_perc.txt",f"document name: {gv.current_file}, text: {sentence}\n","a")
    
    return arabic_count / max(len(non_space_chars),1) >= threshold # return false when ratio < threshold


def word_count_check(sentence):
    if(len(sentence.split()) < 8):
        print_stats("short_sentences_examples.txt",f"document name: {gv.current_file}, text: {sentence}\n","a")
    return len(sentence.split()) >= 8

# remove excessive punctuation marks
def successive_punctuation_check(sentence):
    temp_sentence = re.sub(r'\.\.\.', 'ELLIPSIS', sentence)
    # only removing the excessive punctuation marks
    new_sentence = re.sub(r'([^\w\s\u0600-\u06FF]{4,})', lambda match: match.group(0)[:3], temp_sentence)
    # Restore ellipses
    new_sentence = re.sub('ELLIPSIS', '...', new_sentence)
    # Check if any replacements were made and log the original sentence
    if sentence != new_sentence:
        stats['sentences_modified_due_to_successive_punctuation']+=1
        print_stats("sentences_modified_due_to_successive_punctuation.txt",f"document name: {gv.current_file}\n", "a")
        print_stats("sentences_modified_due_to_successive_punctuation.txt", f"before-> {sentence}\n", "a")
        print_stats("sentences_modified_due_to_successive_punctuation.txt", f"after-> {new_sentence}\n\n", "a")
    return new_sentence


# replace PI with placeholders
def remove_personal_info(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Replace emails with placeholder
    email_count = len(re.findall(email_pattern, text))
    text_with_no_emails = re.sub(email_pattern, "Example@mail.com", text)

    # For phone numbers, we first find them, then replace them one by one
    phone_numbers = list(phonenumbers.PhoneNumberMatcher(text_with_no_emails, None))
    phone_count = len(phone_numbers)

    text_with_no_personal_info = text_with_no_emails
    for match in phone_numbers:
        text_with_no_personal_info = text_with_no_personal_info.replace(match.raw_string, "+999-999-9999")

    if email_count > 0 or phone_count > 0:
        stats['removed_personal_info'] += 1
        print_stats("personal_info_removed.txt", f"document name: {gv.current_file}, text: {text_with_no_personal_info}\n", "a")

    return text_with_no_personal_info

def remove_long_non_arabic_spans(text, threshold=30):
    # Add additional characters (0-9), common punctuation, and parentheses
    allowed_chars = '0-9'
    # Construct the pattern without using re.escape() for the allowed_chars
    pattern = r'[^\u0600-\u06FF' + allowed_chars + r']{' + str(threshold) + r',}'
    stats['sentences_removed_due_to_long_non_arabic_span'] += len(re.findall(pattern, text))
    if(len(re.findall(pattern, text))>0):
        print_stats("sentences_with_long_nonarabic_spans.txt",f"document name: {gv.current_file}\n", "a")
        print_stats("sentences_with_long_nonarabic_spans.txt",f"- before: {text}\n","a")
        text = re.sub(pattern, ' ', text)            
        print_stats("sentences_with_long_nonarabic_spans.txt",f"--- after: {text}\n\n","a")

    return text

def remove_html_js(text):
    try:
        soup = BeautifulSoup(text, features="html.parser")
    except Exception as e:
        # discard the document
        print(f"Error while parsing HTML: {e}")
        text_no_html = ""
        return text_no_html
    # removing js and style from document
    count=0
    for script in soup(["script", "style"]):
        script.extract()
        count+=1
    if count>0:
        stats['documents_cleaned_from_js'] += 1

    # extracting text from html
    try:
        text_no_html = soup.get_text(separator=' ')
    except Exception as e:
        # discard the document
        print(f"Error while parsing HTML: {e}")
        text_no_html = ""
    
    # logging and stats
    if text_no_html != text:
        print_stats("sentences_with_js_html.txt",f"document name: {gv.current_file}\n", "a")
        print_stats("sentences_with_js_html.txt", f"--- before html and specific tag removal: {text}\n\n", "a")
        print_stats("sentences_with_js_html.txt", f"--- after html and specific tag removal: {text_no_html}\n\n", "a")
        stats['documents_cleaned_from_html'] += 1
    return text_no_html

def normalize_text(text):  
    text = re.sub(r'\n+', '\n', text).strip()
    text = re.sub(r'[ \t]+', ' ', text).strip()
    return text

# Normalize arabic text to ensure consistency in its encoding
def normalize_arabic(text):
    return unicodedata.normalize('NFKC', text)

def separate_into_sentences(paragraph):
    # preserves the punctuation at the end of the sentence
    return re.split(r'(?<=\.)\.\.|(?<=[.?!;؟؛])\s+',paragraph)