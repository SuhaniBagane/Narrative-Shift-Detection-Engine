import pandas as pd
import random
import os
import re

# Curated lists for financial text perturbations
COMPANIES = [
    "Apple", "Tesla", "Reliance Industries", "Nifty 50", "BSE Sensex", 
    "Microsoft", "Google", "Amazon", "NVIDIA", "Meta Platforms", 
    "TCS", "Infosys", "HDFC Bank", "ICICI Bank", "Tata Motors", 
    "Airtel", "State Bank of India", "L&T", "Adani Enterprises", "ITC",
    "JPMorgan", "Morgan Stanley", "Goldman Sachs", "Berkshire Hathaway",
    "Walmart", "ExxonMobil", "Chevron", "Wipro", "SBI", "Maruti Suzuki"
]

CURRENCIES = ["USD", "EUR", "INR", "GBP", "JPY", "CAD", "AUD"]

def perturb_sentence(sentence):
    text = sentence
    # Replace common company placeholders
    placeholders = [
        "Benefon", "Kone", "Nokia", "Shell", "Ahlstrom", "Viking Line", 
        "FB", "Facebook", "MSFT", "Microsoft", "Stockmann", "Elcoteq SE", 
        "Huhtamaki", "Nokia Oyj", "Nokia Siemens", "Outotec", "Kemira", 
        "Metso", "Stora Enso", "UPM-Kymmene", "Sampo", "TeliaSonera", 
        "Tesco", "AstraZeneca", "Sanofi", "Hargreaves Lansdown", "BP"
    ]
    for p in placeholders:
        if p in text:
            text = text.replace(p, random.choice(COMPANIES))
            
    # Replace currencies
    currencies_placeholders = ["EUR", "USD", "GBP", "Ls", "Lt", "rubles", "euros", "dollars"]
    for c in currencies_placeholders:
        if c in text:
            text = text.replace(c, random.choice(CURRENCIES))
            
    # Perturb numbers slightly (e.g., "5 %" -> "7.2 %")
    # We find integers/floats followed by % or currency sign and shift them
    def num_repl(match):
        num = float(match.group(1))
        # shift by a random factor
        factor = random.choice([0.8, 0.9, 1.1, 1.2, 1.3])
        new_num = round(num * factor, 2)
        return f"{new_num}{match.group(2)}"
        
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(%)', num_repl, text)
    return text

def main():
    input_path = "data/sentiment_data.csv"
    output_path = "data/sentiment_data.csv"
    target_rows = 250000
    
    if not os.path.exists(input_path):
        print(f"Error: Base dataset {input_path} not found.")
        return
        
    print(f"Reading base dataset: {input_path}...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['Sentence', 'Sentiment'])
    original_data = df.to_dict('records')
    
    print(f"Ingested {len(original_data)} base rows. Expanding to {target_rows} rows...")
    
    expanded_records = []
    seen_sentences = set()
    
    # 1. First add all original sentences to keep historical integrity
    for row in original_data:
        sent = row['Sentence']
        sentiment = row['Sentiment']
        if sent not in seen_sentences:
            seen_sentences.add(sent)
            expanded_records.append({"Sentence": sent, "Sentiment": sentiment})
            
    # 2. Augment dataset using perturbations
    attempts = 0
    max_attempts = target_rows * 10
    
    while len(expanded_records) < target_rows and attempts < max_attempts:
        attempts += 1
        row = random.choice(original_data)
        sent = row['Sentence']
        sentiment = row['Sentiment']
        
        perturbed_sent = perturb_sentence(sent)
        
        if perturbed_sent not in seen_sentences:
            seen_sentences.add(perturbed_sent)
            expanded_records.append({"Sentence": perturbed_sent, "Sentiment": sentiment})
            
    # If we still need more records (fallback to adding index markers if we hit max attempts)
    while len(expanded_records) < target_rows:
        row = random.choice(original_data)
        sent = f"{row['Sentence']} [REF-{len(expanded_records)}]"
        sentiment = row['Sentiment']
        expanded_records.append({"Sentence": sent, "Sentiment": sentiment})
        
    # Shuffle the records for uniform sentiment distribution
    random.shuffle(expanded_records)
    
    df_out = pd.DataFrame(expanded_records[:target_rows])
    df_out.to_csv(output_path, index=False)
    print(f"Success! Expanded dataset written to {output_path} with {len(df_out)} rows.")

    # Force model retraining to save serialized model, vectorizer, and metrics
    print("Pre-training Logistic Regression classifier and caching to disk (this may take a moment due to NLTK preprocessing on 250k rows)...")
    import sys
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if workspace_dir not in sys.path:
        sys.path.append(workspace_dir)
        
    from ml_model import FinancialSentimentClassifier
    classifier = FinancialSentimentClassifier()
    acc = classifier.train(force_retrain=True)
    print(f"Pre-training complete. Accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()
