"""
nlp_utils.py
Shared utilities for NLP PR1: RegEx text cleaning, tokenization,
and domain-specific heuristic sentiment scoring.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Safe NLTK downloads
def ensure_nltk_resources():
    resources = ['stopwords', 'punkt', 'vader_lexicon', 'averaged_perceptron_tagger']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'sentiment/{res}' if res == 'vader_lexicon' else f'taggers/{res}' if 'tagger' in res else f'corpora/{res}')
        except (LookupError, AttributeError):
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
    # Additional checks for newer NLTK versions
    for sub_res in ['punkt_tab', 'averaged_perceptron_tagger_eng']:
        try:
            nltk.download(sub_res, quiet=True)
        except Exception:
            pass

ensure_nltk_resources()

# Custom stop words preserving critical negations
BASE_STOPWORDS = set(stopwords.words('english'))
NEGATION_WORDS = {'no', 'not', 'nor', 'never', 'none', 'neither', 'barely', 'hardly', 'scarcely'}
CUSTOM_STOPWORDS = BASE_STOPWORDS - NEGATION_WORDS

# Domain-specific lexicons for Twitter US Airline Sentiment (35 positive, 35 negative)
POSITIVE_LEXICON = {
    'thanks', 'thank', 'great', 'awesome', 'excellent', 'amazing', 'best', 'good',
    'love', 'loved', 'helpful', 'friendly', 'appreciate', 'appreciated', 'kudos',
    'fantastic', 'wonderful', 'smooth', 'impressed', 'pleasant', 'superb', 'fast',
    'prompt', 'perfect', 'glad', 'safe', 'compliments', 'comfortable', 'nice',
    'kind', 'professional', 'happy', 'on-time', 'hero', 'credit'
}

NEGATIVE_LEXICON = {
    'delay', 'delayed', 'delays', 'cancel', 'cancelled', 'cancelling', 'lost',
    'terrible', 'worst', 'horrible', 'bad', 'poor', 'rude', 'awful', 'stuck',
    'broken', 'disappointed', 'disappointing', 'unacceptable', 'ridiculous',
    'hours', 'wait', 'waiting', 'sucks', 'hold', 'missed', 'damage', 'damaged',
    'useless', 'nightmare', 'stranded', 'hate', 'fail', 'incompetent', 'refund'
}

def clean_text(text: str) -> str:
    """
    Cleans raw tweets using sequential RegEx:
    1. Lowercase text
    2. Remove URLs
    3. Remove @mentions
    4. Handle hashtags (keep token, drop hash)
    5. Strip non-alphanumeric punctuation and numbers
    6. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove @mentions (e.g. @VirginAmerica)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbol but retain word
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove punctuation, symbols, and digits
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_filter(text: str) -> list:
    """
    Tokenizes cleaned text and drops stopwords while retaining negation words.
    """
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in CUSTOM_STOPWORDS and len(w) > 1]
    return filtered

def heuristic_sentiment(text: str) -> dict:
    """
    Heuristic rule-based sentiment classifier with a 2-token negation window.
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    
    pos_score = 0
    neg_score = 0
    
    for i, token in enumerate(tokens):
        # Look back up to 2 words for negation
        is_negated = False
        window_start = max(0, i - 2)
        for prev_token in tokens[window_start:i]:
            if prev_token in NEGATION_WORDS:
                is_negated = True
                break
        
        if token in POSITIVE_LEXICON:
            if is_negated:
                neg_score += 1.0  # Negated positive -> negative
            else:
                pos_score += 1.0
        elif token in NEGATIVE_LEXICON:
            if is_negated:
                pos_score += 1.0  # Negated negative -> positive
            else:
                neg_score += 1.0
                
    if pos_score > neg_score:
        sentiment = 'positive'
    elif neg_score > pos_score:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
        
    return {
        'sentiment': sentiment,
        'pos_score': pos_score,
        'neg_score': neg_score
    }