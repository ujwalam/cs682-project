import random
from transformers import MarianMTModel, MarianTokenizer
import nltk
from nltk.corpus import wordnet2021
import os
import json
import torch

nltk.download('wordnet')

class BackTranslationTraditional():

    def __init__(self, src_lang, tgt_lang) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.en_to_fr_model, self.en_to_fr_tokenizer = \
          self.load_translation_model(
            self.src_lang, 
            self.tgt_lang
        )
        self.fr_to_en_model, self.fr_to_en_tokenizer = \
          self.load_translation_model( 
            self.tgt_lang,
            self.src_lang
        )
        self.device = None

    def load_translation_model(self, src_lang="en", tgt_lang="fr"):
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MarianMTModel.from_pretrained(model_name)
        # model.to(self.device)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def translate_text(self, text, model, tokenizer):
        # inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        # with torch.no_grad():
        #     translated = model.generate(**inputs)
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def back_translation_text(self, text):
        # Translate to French and back to English
        french_text = self.translate_text(
          text, 
          self.en_to_fr_model, 
          self.en_to_fr_tokenizer
        )

        back_translated_text = self.translate_text(
          french_text, 
          self.fr_to_en_model, 
          self.fr_to_en_tokenizer
        )
        return back_translated_text

    def back_translation(self, data):
        back_translated_data = []
        for text in data:
            back_translated_text = self.back_translation_text(text)
            back_translated_data.append(back_translated_text)
        return back_translated_data
    
    def process_data(self, data, output_file, checkpoint_interval=100):
        translated_texts = []
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                translated_text = f.readlines()
            translated_texts = [s.strip() for s in translated_text]

        for idx, text in enumerate(data):
            if idx < len(translated_texts):
                continue

            back_translated = self.back_translation_text(text)
            translated_texts.append(back_translated)

            # Save progress at checkpoints
            if (idx + 1) % checkpoint_interval == 0:
                with open(output_file, 'w') as f:
                    for s in translated_texts:
                        f.write(s + "\n")
                print(f"Checkpoint saved: {idx + 1} sentences processed.")

        # Final save
        with open(output_file, 'w') as f:
            for s in translated_texts:
                f.write(s + "\n")

        print(f"Back translation completed. Total sentences processed: {len(translated_texts)}")












