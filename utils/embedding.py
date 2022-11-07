import numpy as np

class Glove():
    def __init__(self, parameters, ):
        self.file_path = parameters['embedding']['glove']        
        self.embeddings_dict = {}
        with open(self.file_path, 'r', encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector

    def embed(self, word):
        return self.embeddings_dict[word]

