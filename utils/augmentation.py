import os
import numpy as np
import random
import pandas as pd

class DataAugmentation:

    def __init__(self, traditional_bt_file, llm_translated_file, X_train, y_train) -> None:
        self.traditional_translated_texts = []
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_augmented_traditional = []
        self.y_train_augmented_traditional = []
        self.X_train_augmented_llm = []
        self.y_train_augmented_llm = []
        self.X_train_imbalanced = []
        self.y_train_imbalanced = []
        self.seed = None
        if os.path.exists(traditional_bt_file):
            with open(traditional_bt_file, 'r') as f:
                translated_text = f.readlines()
                self.traditional_translated_texts = [s.strip() for s in translated_text]
        with open(llm_translated_file, mode='r') as file:
            self.llm_translated_data = file.read().splitlines()
    
    def augment(self, seed=10):
        indices_dict = {value: np.where(self.y_train == value)[0] for value in np.unique(self.y_train)}
        self.seed = seed
        for label, indices in indices_dict:
            augmented_data_count = random.randint(1, seed)
            augmented_data_indices = random.sample(indices, augmented_data_count)
            original_data_count = seed - augmented_data_count
            original_data_indices = random.sample(indices, original_data_count)
            for idx in augmented_data_indices:
                self.X_train_augmented_traditional.append(self.traditional_translated_texts[idx])
                self.y_train_augmented_traditional.append(label)
                self.X_train_augmented_llm.append(self.llm_translated_data[idx])
                self.y_train_augmented_llm.append(label)
            
            for idx in original_data_indices:
                self.X_train_augmented_traditional.append(self.X_train[idx])
                self.y_train_augmented_traditional.append(label)
                self.X_train_augmented_llm.append(self.X_train[idx])
                self.y_train_augmented_llm.append(label)
                self.X_train_imbalanced.append(self.X_train[idx])
                self.y_train_imbalanced.append(label)
        print(f"Data Augmentation completed. Seed: {seed}")

    def get_llm_augmented_data(self, filename):
        data = pd.DataFrame(self.X_train_augmented_llm, columns=['Text'])
        data['Label'] = self.y_train_augmented_llm
        name = filename + '_' + self.seed + '.csv'
        data.to_csv(name, index=True)
        print(f"LLM Augmented Data successfully written to file : {name}")

    def get_traditional_augmented_data(self, filename):
        data = pd.DataFrame(self.X_train_augmented_traditional, columns=['Text'])
        data['Label'] = self.y_train_augmented_traditional
        name = filename + '_' + self.seed + '.csv'
        data.to_csv(name, index=True)
        print(f"Traditional Augmented Data successfully written to file : {name}")

    def get_original_imbalanced_data(self, filename):
        data = pd.DataFrame(self.X_train_imbalanced, columns=['Text'])
        data['Label'] = self.y_train_imbalanced
        name = filename + '_' + self.seed + '.csv'
        data.to_csv(name, index=True)
        print(f"LLM Augmented Data successfully written to file : {name}")

