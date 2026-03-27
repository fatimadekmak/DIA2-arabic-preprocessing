import re
from num2words import num2words
import torch
from collections import Counter
from eo_pl import TashkeelModel
from tashkeel_tokenizer import TashkeelTokenizer

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def load_model(ckpt_path: str, n_layers: int, max_seq_len: int = 1024):
    tokenizer = TashkeelTokenizer()
    print('ckpt_path is:', ckpt_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    print('Creating Model...')
    model = TashkeelModel(tokenizer, max_seq_len=max_seq_len, n_layers=n_layers, learnable_pos_emb=False)
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval().to(device)
    return model 

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

DIACRITIC_PATTERN = re.compile(
    '['
    '\u064B-\u0652'  #  ً ٌ ٍ َ ُ ِّ ْ
    '\u0670'         #  ٰ
    ']'
)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

FATHATAN = u'\u064b'
DAMMATAN = u'\u064c'
KASRATAN = u'\u064d'
FATHA = u'\u064e'
DAMMA = u'\u064f'
KASRA = u'\u0650'
SHADDA = u'\u0651'
SUKUN = u'\u0652'

HARAKAT_PAT = re.compile(u"["+u"".join([FATHATAN, DAMMATAN, KASRATAN,
                                        FATHA, DAMMA, KASRA, SUKUN,
                                        SHADDA])+u"]")

def strip_tashkeel(text):
    text = HARAKAT_PAT.sub('', text)
    text = re.sub(u"[\u064E]", "", text,  flags=re.UNICODE) # fattha
    text = re.sub(u"[\u0671]", "", text,  flags=re.UNICODE) # waSla
    return text

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# Pattern to split on punctuation, keeping them as separate tokens
PUNC_PATTERN = r'(\s*[،\.؟!؛:?,;\n]\s*)'

def split_by_punc_keep(text: str) -> list:
    """
    Split `text` on punctuation, returning a list where punctuation tokens are at odd indices.
    """
    return re.split(PUNC_PATTERN, text)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def split_long_segment(text: str, max_len: int) -> list:
    """
    Further split a long segment into sub-segments ≤ max_len, breaking at word boundaries.
    If a single word exceeds max_len, hard-break the word.
    """
    words = text.split()
    sub_segs = []
    current = ""
    for w in words:
        cand = f"{current} {w}" if current else w
        if len(cand) > max_len:
            if current:
                sub_segs.append(current)
            if len(w) > max_len:
                # hard-break oversized word
                for i in range(0, len(w), max_len):
                    sub_segs.append(w[i:i+max_len])
                current = ""
            else:
                current = w
        else:
            current = cand
    if current:
        sub_segs.append(current)
    return sub_segs

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def diacritize(text: str, model) -> str:
    batch_size = 1
    verbose = False
    assert len(text) <= 1024
    x_tashkeel = model.do_tashkeel_batch([text], batch_size, verbose)
    return x_tashkeel[0]  

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def numeric_expansion_and_record(raw: str):
    """
    - Finds every integer token in 'raw'.
    - Replaces it with num2words(..., lang='ar'), and records:
        (start_index_in_X1, end_index_in_X1, integer_value)
    Returns:
      X1    = the text after replacing digits → Arabic words (no diacritics yet),
      spans = a list of tuples (start_j, end_j, numeric_value).
    """
    spans = []
    pieces = []
    last_idx = 0
    digit_re = re.compile(r'(?<!\d)\d+(?!\d)')
    for m in digit_re.finditer(raw):
        start_raw, end_raw = m.span()
        val = int(m.group())
        arabic_words = num2words(val, lang='ar')
        arabic_words = strip_tashkeel(arabic_words)
        pieces.append(raw[last_idx:start_raw])
        # Record where the arabic_words will live in X1:
        new_start = sum(len(p) for p in pieces)
        pieces.append(arabic_words)
        if(any(char.isdigit() for char in "".join(pieces))==True):
            print("numbers remain") 
        new_end = new_start + len(arabic_words)
        spans.append((new_start, new_end, val))
        last_idx = end_raw
        

    # Append the remainder
    pieces.append(raw[last_idx:])
    X1 = "".join(pieces)
    return X1, spans

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def symbol_expansion_and_record(raw: str):
    symbol_map = {'%': 'في المئة', '$': 'دولار'}
    spans = []
    pieces = []
    last_idx = 0

    for m in re.finditer(r'[%$]', raw):
        start, end = m.span()
        sym = m.group()
        word = symbol_map[sym]
        # original text up to the symbol
        before = raw[last_idx:start]
        pieces.append(before)
        # decide whether to prefix a space
        if before.endswith(' '):
            expanded = word
        else:
            expanded = ' ' + word
        # record the span in the newly built text
        new_start = sum(len(p) for p in pieces)
        pieces.append(expanded)
        new_end = new_start + len(expanded)
        spans.append((new_start, new_end, sym))
        last_idx = end
    # append the rest of the text
    pieces.append(raw[last_idx:])
    return ''.join(pieces), spans

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def build_index_mapping(raw: str, X1: str, X2: str):
    """
    Build a dict `mapping` so that for every index j in X1 (a base-letter or punctuation
    or space), mapping[j] = the index i in X2 of the *corresponding base-letter*.
    We walk through X2, skipping any combining diacritic codepoints. Whenever we see
    a non-diacritic in X2, it must match X1[j]. We record mapping[j] = i, then j+=1, i+=1.
    """
    mapping = {}
    i = 0
    j = 0
    len_X1 = len(X1)
    len_X2 = len(X2)

    while j < len_X1 and i < len_X2:
        # If X2[i] is a combining diacritic, skip it.
        if DIACRITIC_PATTERN.match(X2[i]):
            i += 1
            continue
        if DIACRITIC_PATTERN.match(X1[j]):
            # If X1[j] is a diacritic, skip it.
            j += 1
            continue
        # Now X2[i] should be the same base-letter or punctuation or space as X1[j].
        if X2[i] != X1[j]:
            with open("error_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Base-letter mismatch at X1[{j}]='{X1[j]}' vs X2[{i}]='{X2[i]}''.\n")
                f.write(f"Raw text: {raw}\n")
                f.write(f"X1: {X1}\n")
                f.write(f"X2: {X2}\n\n")
            raise ValueError(
                f"Base-letter mismatch at X1[{j}]='{X1[j]}' vs X2[{i}]='{X2[i]}''. "
            )
        mapping[j] = i
        j += 1
        i += 1

    if j != len_X1:
        raise ValueError(
            f"Not all characters in X1 mapped (j={j} of {len_X1})."
        )
    return mapping

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def restore_digits_after_diacritization(X2: str, spans, mapping):
    """
    Given X2 (the diacritized string), the list `spans` = [(start_j, end_j, val), ...],
    and the mapping from X1-indices → X2-indices, replace each diacritized-number span
    with its original integer digits.

    We do replacements in descending order of start_j so that earlier indices stay valid.
    """
    # Sort spans by start_j descending
    spans_desc = sorted(spans, key=lambda x: x[0], reverse=True)
    result = X2

    for (start_j, end_j, val) in spans_desc:
        # Find in X2: start_i = mapping[start_j]
        start_i = mapping[start_j]

        # The last base letter in X1's span is at index end_j - 1. Its mapped index in X2:
        last_j = end_j - 1
        last_i = mapping[last_j]

        # We must also include _all_ diacritics immediately after last_i in X2:
        end_i = last_i + 1
        while end_i < len(result) and DIACRITIC_PATTERN.match(result[end_i]):
            end_i += 1

        # Now `result[start_i:end_i]` is the full diacritized phrase. Replace it:
        result = result[:start_i] + str(val) + result[end_i:]

        # We do not update `mapping` here, because we do these replacements from right→left,
        # on the working copy `result`. No earlier span's indices in X2 are affected.

    return result

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def restore_symbols_after_diacritization(text: str, symbol_spans, mapping) -> str:
    """
    Given a diacritized string `text`, symbol_spans = [(start_j, end_j, sym), ...]
    and mapping from X1-indices to text indices, restore each original symbol `sym`
    into `text` by replacing the span of words at mapping[start_j:end_j] with `sym`.
    """
    result = text
    for (start_j, end_j, sym) in sorted(symbol_spans, key=lambda x: x[0], reverse=True):
        # find start_i in result
        start_i = mapping[start_j]
        # find last base char at end_j-1
        last_i = mapping[end_j - 1]
        # include any following diacritics
        end_i = last_i + 1
        while end_i < len(result) and DIACRITIC_PATTERN.match(result[end_i]):
            end_i += 1
        # splice in the symbol
        result = result[:start_i] + sym + result[end_i:]
    return result

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

TASHKEEL = re.compile(r'[\u064B-\u0652]')
TATWEEL = '\u0640'
NON_ARABIC = re.compile(r'[^\u0621-\u063A\u0641-\u064A\u0660-\u0669 0-9 \n،\.؟!؛:?%$,;]')

def remove_non_arabic_modified(text):
    removed = {
        'tashkeel': Counter(TASHKEEL.findall(text)),
        'tatweel': Counter(c for c in text if c == TATWEEL),
    }
    text = TASHKEEL.sub('', text).replace(TATWEEL, '')
    removed['non_arabic'] = Counter(NON_ARABIC.findall(text))
    text = ' '.join(NON_ARABIC.sub(' ', text).split(" "))
    return text, removed

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def normalize_non_core_arabic(text: str) -> str:
    substitutions = {
        'ڤ': 'ف',
        'چ': 'ج',
        'گ': 'ك',
        'پ': 'ب',
        'ۀ': 'ه',
    }
    for k, v in substitutions.items():
        text = text.replace(k, v)
    return text

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    max_seq_len = 1024
    model = load_model(max_seq_len)

    raw = "يعي بين 262 و264 264 هو عدد صحيح طبيعي بين 263 و 265 265 هو عدد صحيح طبيعي بين 264 و 266 266 هو عدد"

