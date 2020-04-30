import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import json
from typing import List

from config.hparams import *

class Vocabulary(object):
  """
     A simple Vocabulary class which maintains a mapping between words and integer tokens. Can be
     initialized either by word counts from the VisDial v1.0 train dataset, or a pre-saved
     vocabulary mapping.

     Parameters
     ----------
     word_counts_path: str
         Path to a json file containing counts of each word across captions, questions and answers
         of the VisDial v1.0 train dataset.
     min_count : int, optional (default=0)
         When initializing the vocabulary from word counts, you can specify a minimum count, and
         every token with a count less than this will be excluded from vocabulary.
   """

  PAD_TOKEN = "<PAD>"
  SOS_TOKEN = "<S>"
  EOS_TOKEN = "</S>"
  UNK_TOKEN = "<UNK>"

  PAD_INDEX = 0
  SOS_INDEX = 1
  EOS_INDEX = 2
  UNK_INDEX = 3

  def __init__(self, word_counts_path: str, min_count: int = 5):
    if not os.path.exists(word_counts_path):
      raise FileNotFoundError(f"Word counts do not exist at {word_counts_path}")

    with open(word_counts_path, "r") as word_counts_file:
      word_counts = json.load(word_counts_file)

      # form a list of (word, count) tuples and apply min_count threshold
      word_counts = [
        (word, count) for word, count in word_counts.items() if count >= min_count
      ]
      # sort in descending order of word counts
      word_counts = sorted(word_counts, key=lambda wc: -wc[1])
      words = [w[0] for w in word_counts]

    self.word2index = {}
    self.word2index[self.PAD_TOKEN] = self.PAD_INDEX
    self.word2index[self.SOS_TOKEN] = self.SOS_INDEX
    self.word2index[self.EOS_TOKEN] = self.EOS_INDEX
    self.word2index[self.UNK_TOKEN] = self.UNK_INDEX
    for index, word in enumerate(words):
      self.word2index[word] = index + 4

    self.index2word = {index: word for word, index in self.word2index.items()}

  @classmethod
  def from_saved(cls, saved_vocabulary_path: str) -> "Vocabulary":
    """Build the vocabulary from a json file saved by ``save`` method.

    Parameters
    ----------
    saved_vocabulary_path : str
        Path to a json file containing word to integer mappings (saved vocabulary).
    """
    with open(saved_vocabulary_path, "r") as saved_vocabulary_file:
      cls.word2index = json.load(saved_vocabulary_file)
    cls.index2word = {index: word for word, index in cls.word2index.items()}

  def to_indices(self, words: List[str]) -> List[int]:
    return [self.word2index.get(word, self.UNK_INDEX) for word in words]

  def to_words(self, indices: List[int]) -> List[str]:
    return [self.index2word.get(index, self.UNK_TOKEN) for index in indices]

  def save(self, save_vocabulary_path: str) -> None:
    with open(save_vocabulary_path, "w") as save_vocabulary_file:
      json.dump(self.word2index, save_vocabulary_file)

  def __len__(self):
    return len(self.index2word)


class GloveProcessor(object):
  def __init__(self, glove_path):
    self.glove_path = glove_path

  def _load_glove_model(self):
    print("Loading pretrained word vectors...")
    with open(self.glove_path, 'r') as f:
      model = {}
      for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])  # e.g., 300 dimension
        model[word] = embedding

    print("Done.", len(model), " words loaded from %s" % self.glove_path)

    return model

  def save_glove_vectors(self, vocabulary, glove_npy_path, dim=300):
    """
    Saves glove vectors in numpy array
    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    # vocabulary index2word
    vocab_size = len(vocabulary.index2word)
    glove_embeddings = self._load_glove_model()
    embeddings = np.zeros(shape=[vocab_size, 300], dtype=np.float32)

    vocab_in_glove = 0
    for i in range(0, vocab_size):
      word = vocabulary.index2word[i]
      if word in ['<PAD>', '<S>', '</S>']:
        continue
      if word in glove_embeddings:
        embeddings[i] = glove_embeddings[word]
        vocab_in_glove += 1
      else:
        embeddings[i] = glove_embeddings['unk']

    print("Vocabulary in GLoVE : %d / %d" % (vocab_in_glove, vocab_size))
    np.save(glove_npy_path, embeddings)


if __name__ == '__main__':
  hparams = BASE_PARAMS

  vocabulary = Vocabulary(hparams["word_counts_json"], hparams["vocab_min_count"])
  print("### The number of train vocabulary : %d" % len(vocabulary.word2index))

  glove_vocab = GloveProcessor(hparams["pretrained_glove"])
  glove_vocab.save_glove_vectors(vocabulary, hparams["glove_npy"])