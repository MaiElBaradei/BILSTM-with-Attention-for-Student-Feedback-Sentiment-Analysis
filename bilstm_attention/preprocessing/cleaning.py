import re
from typing import Set

def clean_text(text: str) -> str:
    """
    Applies standardized lowercasing, HTML removal, and character normalization.
    Args:
        text (str): The text to clean.
    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_stopwords(lang: str = "english") -> Set[str]:
    """Return NLTK stopwords, downloading the corpus on first use if needed."""
    try:
        import nltk
        from nltk.corpus import stopwords as _sw
        try:
            return set(_sw.words(lang))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(_sw.words(lang))
    except ImportError:
        # Fallback dictionary
        return {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
            "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
            "but", "by", "can", "can't", "cannot", "com", "could", "couldn't", "did", "didn't", "do", 
            "does", "doesn't", "doing", "don't", "down", "during", "each", "else", "even", "ever", "every", 
            "few", "for", "from", "further", "get", "got", "had", "hadn't", "has", "hasn't", "have", 
            "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", 
            "him", "himself", "his", "how", "how's", "however", "i", "i'd", "i'll", "i'm", "i've", "if", 
            "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just", "let's", "like", "ll", "lot", 
            "made", "make", "many", "may", "me", "might", "mine", "more", "most", "mustn't", "my", "myself", 
            "needn't", "never", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", 
            "ought", "our", "ours", "ourselves", "out", "over", "own", "re", "same", "shan't", "she", "she'd", 
            "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", 
            "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", 
            "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", 
            "us", "ve", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", 
            "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
            "why", "why's", "will", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", 
            "you've", "your", "yours", "yourself", "yourselves"
        }