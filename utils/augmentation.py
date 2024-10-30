from collections import Counter
import os
import numpy as np
import random

class DataAugmentation:

    def __init__(self, traditional_bt_file, llm_translated_data, X_train, y_train) -> None:
        self.traditional_translated_texts = []
        self.X_train = X_train
        self.y_train = y_train
        self.llm_translated_data = llm_translated_data
        self.X_train_augmented_traditional = []
        self.y_train_augmented_traditional = []
        self.X_train_augmented_llm = []
        self.y_train_augmented_llm = []
        if os.path.exists(traditional_bt_file):
            with open(traditional_bt_file, 'r') as f:
                translated_text = f.readlines()
            traditional_translated_texts = [s.strip() for s in translated_text]
    
    def augment(self, seed = 10):
        indices_dict = {value: np.where(self.y_train == value)[0] for value in np.unique(self.y_train)}
        for label, indices in indices_dict:
            random_indices = random.sample(indices, seed)
            augmented_data_count = random.randint(1, seed)
            augmented_data_indices = random.sample(indices, augmented_data_count)
            original_data_count = seed - augmented_data_count
            original_data_indices = random.sample(indices, original_data_count)
            for idx in augmented_data_indices:
                self.X_train_augmented_traditional.append(self.traditional_translated_texts[idx])
                self.y_train_augmented_traditional.append(label)
                self.X_train_augmented_llm.append(self.llm_translated_data[idx])
                self.y_train_augmented_traditional.append(label)
            
            for idx in original_data_indices:
                self.X_train_augmented_traditional.append(self.X_train[idx])
                self.y_train_augmented_traditional.append(label)
                self.X_train_augmented_llm.append(self.X_train[idx])
                self.y_train_augmented_traditional.append(label)

    def get_llm_augmented_data(self):
        return self.X_train_augmented_llm, self.y_train_augmented_llm 

    def get_traditional_augmented_data(self):
        return self.X_train_augmented_traditional, self.y_train_augmented_traditional 

