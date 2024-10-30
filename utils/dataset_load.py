from datasets import load_dataset


class DatasetLoader:
  def __init__(self, dataset_name):
    self.dataset_name = dataset_name
    self.dataset = {}
    self.X_train = None
    self.y_train = None
    self.X_test = None
    self.y_test = None

  def load(self):

    if self.dataset_name == 'banking77':
      self.dataset = load_dataset("polyai/banking77", trust_remote_code=True)
      self.X_train = self.dataset['train']['text']
      self.y_train = self.dataset['train']['label']
      self.X_test = self.dataset['test']['text']
      self.y_test = self.dataset['test']['label']
      print(f"{self.dataset_name} dataset loaded successfully.")
      
    else:
        raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
  
    return self.X_train, self.y_train, self.X_test, self.y_test
  
  def restart_load(self):
      return self.X_train, self.y_train, self.X_test, self.y_test
