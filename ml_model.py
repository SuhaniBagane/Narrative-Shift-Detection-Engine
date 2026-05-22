"""
BuzzStreet – ml_model.py
Machine Learning Module.
Handles: Labeled training dataset generation, TF-IDF vectorization,
Logistic Regression training, model evaluation (accuracy, confusion matrix, classification report),
and sentiment probability/label prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nlp_pipeline import preprocess_text

# A curated labeled dataset of financial sentiment headlines for on-the-fly training
LABELED_DATA = [
    # Positive Sentiments
    ("Quarterly profits surged by 45% exceeding all street expectations.", "Positive"),
    ("New product launch is an absolute success, boosting company stock.", "Positive"),
    ("Stocks rally as trade tensions begin to ease between major economic powers.", "Positive"),
    ("Company secures a massive $500 million defense contract.", "Positive"),
    ("Revenue climbs to record high as e-commerce segment doubles in size.", "Positive"),
    ("Analyst upgrades stock rating to strong buy following impressive earnings.", "Positive"),
    ("Merger approved, creating the largest player in the renewable energy sector.", "Positive"),
    ("Consumer confidence index rises, signaling strong retail growth ahead.", "Positive"),
    ("Tech giant announces a huge $10 billion stock buyback program.", "Positive"),
    ("Unemployment rates drop to a ten-year low, boosting economic outlook.", "Positive"),
    ("Biotech firm shares jump 30% after successful phase 3 clinical trial.", "Positive"),
    ("Dividend payout increased by 15% due to strong free cash flow.", "Positive"),
    ("Inflation cooling faster than expected, sparking market enthusiasm.", "Positive"),
    ("Logistics company reports complete resolution of supply chain issues.", "Positive"),
    ("Venture capital investments in AI startups hit record highs.", "Positive"),
    ("S&P 500 reaches a new record high, driven by technology and banking stocks.", "Positive"),
    ("Corporate tax cuts expected to boost business investment and jobs.", "Positive"),
    ("Automaker EV sales double in the first half of the year.", "Positive"),
    ("FDI inflows surge as domestic markets show exceptional resilience.", "Positive"),
    ("Startup receives major funding round from top tier venture firms.", "Positive"),
    ("Exports jump to an all time high due to rising global demand.", "Positive"),
    ("Corporate earnings showcase stellar resilience across retail and finance.", "Positive"),
    ("Infrastructure spending bill approved by congress, boosting building sector.", "Positive"),
    ("Acquisition of rival firm expected to deliver massive synergy benefits.", "Positive"),
    ("Factory output expands at the fastest pace in over two years.", "Positive"),
    
    # Negative Sentiments
    ("Company reports sharp decline in quarterly profits, shares slide 12%.", "Negative"),
    ("Inflation spikes to a 40-year high, triggering immediate interest rate hikes.", "Negative"),
    ("Fears of a global recession grow as industrial output contracts.", "Negative"),
    ("Supply chain disruptions halt production at major automotive manufacturing plants.", "Negative"),
    ("Retail giant files for bankruptcy protection after accumulating severe debts.", "Negative"),
    ("Regulators launch antitrust lawsuit against leading tech conglomerate.", "Negative"),
    ("Severe drought leads to widespread crop failures, driving agricultural prices up.", "Negative"),
    ("Major cyberattack exposes customer financial data, causing shares to plunge.", "Negative"),
    ("Corporate bond defaults hit multi-year highs, raising credit risk concerns.", "Negative"),
    ("Housing market shows signs of collapse as mortgage defaults surge.", "Negative"),
    ("Oil prices skyrocket past $100, threatening corporate margins and consumer spending.", "Negative"),
    ("Unemployment claims rise unexpectedly, indicating weakening labor market.", "Negative"),
    ("Rating agency downgrades government credit rating, sparking currency drop.", "Negative"),
    ("Pharma stock collapses 40% after key drug fails FDA approval.", "Negative"),
    ("Fiscal deficit widens dramatically, raising concerns over economic stability.", "Negative"),
    ("European indices sink following unexpected political instability.", "Negative"),
    ("Central bank warns of economic slowdown in the coming quarters.", "Negative"),
    ("Consumer spending retracts as high living costs pinch households.", "Negative"),
    ("Global semiconductor shortage continues to throttle production lines.", "Negative"),
    ("Company cuts dividend payout by 50% to conserve capital.", "Negative"),
    ("Trade tariffs set to hurt domestic manufacturing cost efficiency.", "Negative"),
    ("Investor panic rises as tech sector valuation bubble threatens to burst.", "Negative"),
    ("Bank runs reported at regional institutions, raising liquidity fears.", "Negative"),
    ("Corporate earnings disappoint across the board in Q3 reports.", "Negative"),
    ("Geopolitical tensions escalate, threatening key maritime trade routes.", "Negative"),
    
    # Neutral Sentiments
    ("Federal Reserve is scheduled to meet next week to discuss monetary policy.", "Neutral"),
    ("Automobile manufacturers are shifting resources to develop electric models.", "Neutral"),
    ("The municipal department announces new zoning laws for industrial zones.", "Neutral"),
    ("Global trade volume remained flat throughout the second quarter.", "Neutral"),
    ("Tech companies adapt to the newly introduced internet privacy regulations.", "Neutral"),
    ("Consumer preferences show a gradual drift toward organic packaging.", "Neutral"),
    ("Minimum wage adjustments are being debated in several local assemblies.", "Neutral"),
    ("Energy usage patterns have stabilized with standard hybrid working hours.", "Neutral"),
    ("Real estate transaction volumes plateaued after a period of rapid growth.", "Neutral"),
    ("Analysts are waiting for the upcoming quarterly earnings release reports.", "Neutral"),
    ("New regulatory compliance steps are proposed for the gig economy workforce.", "Neutral"),
    ("Demographic shifts are slowly influencing long term commercial market trends.", "Neutral"),
    ("International travel corridors have reopened to pre pandemic schedules.", "Neutral"),
    ("Several central banks are researching the utility of sovereign digital tokens.", "Neutral"),
    ("The revised tax filing rules have officially come into effect this fiscal term.", "Neutral"),
    ("Commodities trading showed typical seasonal trends during the harvest cycle.", "Neutral"),
    ("Annual financial reports for state run public sector undertakings were tabled.", "Neutral"),
    ("The central statistics office will publish the quarterly GDP estimates on Friday.", "Neutral"),
    ("Company enters routine wage negotiations with its labor union.", "Neutral"),
    ("New CEO outlines long-term strategy in a routine shareholder address.", "Neutral"),
    ("Standard regulatory review of the telecom merger is underway.", "Neutral"),
    ("Retail inventory levels remain steady ahead of the autumn season.", "Neutral"),
    ("Bond yields fluctuate within a narrow range during morning trading.", "Neutral"),
    ("State government announces study on renewable energy integration.", "Neutral"),
    ("Industry association holds annual summit in the capital city.", "Neutral")
]

# Duplicate the dataset with small keyword variations to create a robust training corpus of ~150 samples
def get_expanded_training_corpus():
    expanded = []
    # Base corpus
    for text, label in LABELED_DATA:
        expanded.append((text, label))
        
    # Variations for positive
    pos_variations = [
        ("Stock market surges as investor confidence rebounds on positive news.", "Positive"),
        ("Strong quarterly results push market index to new record heights.", "Positive"),
        ("Tech startups secure record venture capital funding this quarter.", "Positive"),
        ("Merger creates major renewable energy powerhouse, stocks jump.", "Positive"),
        ("Outstanding earnings report from banking sector drives market gains.", "Positive"),
        ("FDI inflows grow by 20%, showing immense confidence in economy.", "Positive"),
        ("New product sales break records in opening week.", "Positive"),
        ("Analysts upgrade IT sector stocks, citing robust international demand.", "Positive"),
        ("Key industrial indices post sharp gains after factory data expands.", "Positive")
    ]
    # Variations for negative
    neg_variations = [
        ("Market declines heavily as inflation fears trigger widespread selloffs.", "Negative"),
        ("Central bank interest rate hike triggers panic selling in bonds and stocks.", "Negative"),
        ("Severe supply chain bottlenecks throttle factory output globally.", "Negative"),
        ("Debt crisis deepens as major real estate developer defaults on bonds.", "Negative"),
        ("Quarterly earnings miss triggers massive retail sector stock dump.", "Negative"),
        ("Unemployment claims jump to highest levels in six months.", "Negative"),
        ("Cybersecurity breach compromises financial data, stocks sink.", "Negative"),
        ("Credit downgrade triggers sharp selloff in government bonds and currency.", "Negative"),
        ("Tech giant faces heavy fines following antitrust regulatory probe.", "Negative")
    ]
    # Variations for neutral
    neu_variations = [
        ("Investors remain cautious ahead of central bank policy announcement.", "Neutral"),
        ("GDP growth figures to be released by government stats office tomorrow.", "Neutral"),
        ("Major industries adapt gradually to new environmental guidelines.", "Neutral"),
        ("Consumer behavior remains steady with standard seasonal purchasing.", "Neutral"),
        ("Gold prices trade in a tight consolidative pattern today.", "Neutral"),
        ("Company schedules annual general meeting of shareholders for next month.", "Neutral"),
        ("Regulators examine the proposed merger for compliance.", "Neutral"),
        ("Retail market prices stabilize after recent regulatory adjustments.", "Neutral"),
        ("Bond markets trade flat amidst quiet trading sessions.", "Neutral")
    ]
    
    for text, label in pos_variations + neg_variations + neu_variations:
        expanded.append((text, label))
        
    # Add another round of synthetic expansions to ensure solid training set size (~150-180 samples)
    syn_pos = [
        ("Stellar corporate performance drives heavy institutional buying.", "Positive"),
        ("Inflation cooling down sparks massive rally in housing and energy stocks.", "Positive"),
        ("Venture funding hits record high in tech sector as investors rejoice.", "Positive"),
        ("Excellent trade surplus figures lift domestic currency against dollar.", "Positive"),
        ("Record tourist footfall boosts hospitality and travel sectors.", "Positive"),
        ("Successful commercial rollout of 5G infrastructure boosts telecom shares.", "Positive")
    ]
    syn_neg = [
        ("Liquidity crunch forces investment firm to freeze redemptions, panic ensues.", "Negative"),
        ("Surging wholesale prices spark panic over persistent hyperinflation.", "Negative"),
        ("Automobile manufacturing drops sharply due to global chip crisis.", "Negative"),
        ("Bankruptcy filings jump by 30% as corporate debt costs double.", "Negative"),
        ("Geopolitical gridlock stops key maritime oil tankers, sending energy prices up.", "Negative"),
        ("Major crop failures drive global food inflation to worrying heights.", "Negative")
    ]
    syn_neu = [
        ("Federal committee to announce tax rate decisions in autumn session.", "Neutral"),
        ("Retail stores observe normal footfall in standard quarterly review.", "Neutral"),
        ("Tech sector adjusts quietly to recent privacy policy changes.", "Neutral"),
        ("Central bank holds standard treasury auction with stable yield outcomes.", "Neutral"),
        ("Agricultural yield estimates remain unchanged from spring projections.", "Neutral"),
        ("Energy consumption maps normal curve during summer holiday period.", "Neutral")
    ]
    
    for text, label in syn_pos + syn_neg + syn_neu:
        expanded.append((text, label))
        
    return expanded


class FinancialSentimentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        # Use multinomial logistic regression with small regularization C
        self.model = LogisticRegression(C=2.0, solver='lbfgs', max_iter=500)
        self.accuracy = 0.0
        self.report = ""
        self.confusion_matrix = None
        self.is_trained = False
        
    def train(self):
        """
        Loads the training corpus (including CSV dataset if available), cleans text using the nlp_pipeline,
        extracts TF-IDF features, and trains the Logistic Regression classifier.
        Saves accuracy and evaluation metrics.
        """
        corpus = get_expanded_training_corpus()
        
        # Load from CSV if it exists
        import os
        csv_path = "data/sentiment_data.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'Sentence' in df.columns and 'Sentiment' in df.columns:
                    df = df.dropna(subset=['Sentence', 'Sentiment'])
                    df['Sentiment'] = df['Sentiment'].str.strip().str.capitalize()
                    # Filter valid labels
                    df = df[df['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
                    csv_data = list(zip(df['Sentence'], df['Sentiment']))
                    corpus.extend(csv_data)
            except Exception as e:
                print(f"Warning: Failed to load training CSV {csv_path}: {e}")
        
        # Preprocess all training texts
        cleaned_texts = [preprocess_text(text) for text, _ in corpus]
        labels = [label for _, label in corpus]
        
        # Fit vectorizer and transform text
        tfidf_features = self.vectorizer.fit_transform(cleaned_texts)
        
        # Train-Test Split (80% train, 20% test) to generate realistic project metrics
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Fit Model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.report = classification_report(y_test, y_pred, output_dict=True)
        self.confusion_matrix = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])
        self.is_trained = True
        
        return self.accuracy

    def predict_sentiment_lr(self, raw_headline):
        """
        Predicts the sentiment probability distribution (Negative, Neutral, Positive)
        and final label for a raw headline.
        """
        if not self.is_trained:
            self.train()
            
        cleaned = preprocess_text(raw_headline)
        vectorized = self.vectorizer.transform([cleaned])
        
        # Get class label prediction
        label = self.model.predict(vectorized)[0]
        
        # Get probability distributions
        probs = self.model.predict_proba(vectorized)[0]
        classes = self.model.classes_ # array of ['Negative', 'Neutral', 'Positive'] in alphabetical order
        
        prob_dict = dict(zip(classes, probs))
        
        # Map values consistently
        neg_score = float(prob_dict.get("Negative", 0.0))
        neu_score = float(prob_dict.get("Neutral", 0.0))
        pos_score = float(prob_dict.get("Positive", 0.0))
        
        return {
            "Positive": pos_score,
            "Negative": neg_score,
            "Neutral": neu_score,
            "Sentiment Label": label
        }

    def get_top_tfidf_features(self, headline, top_n=5):
        """
        Extracts the highest TF-IDF features and their weights for a given headline.
        """
        cleaned = preprocess_text(headline)
        vectorized = self.vectorizer.transform([cleaned])
        
        feature_names = self.vectorizer.get_feature_names_out()
        scores = vectorized.toarray()[0]
        
        # Pair feature names with their TF-IDF scores
        feature_scores = [(feature_names[i], scores[i]) for i in range(len(scores)) if scores[i] > 0]
        # Sort by score descending
        feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
        
        return feature_scores[:top_n]

# Singleton instance of the classifier
model_instance = FinancialSentimentClassifier()
model_instance.train() # Train on startup
