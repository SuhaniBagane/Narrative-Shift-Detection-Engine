"""
BuzzStreet – nlp_pipeline.py
Advanced NLP Preprocessing Pipeline.
Implements: lowercasing, noise removal, tokenization, stopword removal,
and POS-tagged lemmatization with automatic NLTK resource downloading.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Lazy downloader for NLTK resources to prevent startup delays
def ensure_nltk_resources():
    required_packages = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]
    for resource, package in required_packages:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)

# Run downloader on module import
ensure_nltk_resources()

def get_wordnet_pos(treebank_tag):
    """
    Maps NLTK Treebank POS tags to WordNet POS tags
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_headline_detailed(text):
    """
    Performs complete NLP preprocessing on a single headline and returns
    a dictionary containing each intermediate step's output for demonstration.
    """
    ensure_nltk_resources()
    
    # 1. Original
    steps = {"original": text}
    
    # 2. Lowercase
    lowercased = text.lower()
    steps["lowercase"] = lowercased
    
    # 3. Noise Removal (strip URLs, HTML tags, emails, non-alphanumeric except percentage/currency)
    # Remove URLs
    no_urls = re.sub(r'https?://\S+|www\.\S+', '', lowercased)
    # Remove HTML tags
    no_html = re.sub(r'<.*?>', '', no_urls)
    # Remove general special characters, keeping alphanumeric and basic currency/percentage
    clean_noise = re.sub(r'[^a-zA-Z0-9\s%$\-]', '', no_html)
    steps["noise_removed"] = clean_noise
    
    # 4. Tokenization
    tokens = word_tokenize(clean_noise)
    steps["tokenized"] = tokens.copy()
    
    # 5. Stopword Removal
    stop_words = set(stopwords.words('english'))
    # Keep some critical financial words if they are in stopwords list (though usually not)
    no_stopwords = [w for w in tokens if w not in stop_words]
    steps["stopwords_removed"] = no_stopwords.copy()
    
    # 6. Part-of-Speech Tagging
    try:
        pos_tags = nltk.pos_tag(no_stopwords)
    except Exception:
        # Fallback if POS tagger model fails
        pos_tags = [(w, 'N') for w in no_stopwords]
    steps["pos_tags"] = pos_tags
    
    # 7. Lemmatization using POS Tags
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, wn_tag)
        lemmas.append(lemma)
        
    steps["lemmatized"] = lemmas.copy()
    
    # 8. Final Cleaned Text Representation
    steps["final_text"] = " ".join(lemmas)
    
    return steps

def preprocess_text(text):
    """
    Simple wrapper returning only the final cleaned string.
    """
    return preprocess_headline_detailed(text)["final_text"]
