import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

class DataPreprocessing:
    def __init__(self,language):
        self.language = language
    
    def preprocess_text(self, text):
        # Tokenization
        tokens = word_tokenize(text)
    
        # Remove stop words
        stop_words = set(stopwords.words(self.language))
        filtered_tokens = [word for word in tokens if word not in stop_words]
    
        # Stemming or lemmatization
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
        # Join tokens back into a string
        preprocessed_text = ' '.join(stemmed_tokens)
    
        return preprocessed_text
    
    def preprocess_data(self, text_array):
        processed_data = []
        for text in text_array:
            processed_text = self.preprocess_text(text)
            processed_data.append(processed_text)
        return processed_data