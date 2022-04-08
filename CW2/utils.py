import itertools
import torch
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from vocabulary import Vocabulary
from config import *


def read_lines(filepath):
    """ Open the ground truth captions into memory, line by line. 
    Args:
        filepath (str): the complete path to the tokens txt file
    """
    file = open(filepath, 'r')
    lines = []

    while True: 
        # Get next line from file 
        line = file.readline() 
        if not line: 
            break
        lines.append(line.strip())
    file.close()
    return lines


def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of lists of str): lists of words
    """
    image_ids = []
    cleaned_captions = []

    for line in lines:
        # Separates the image ids and the captions.
        image_id, caption = line.split('\t')

        # Removes punctuation, numbers, and symbols from captions and converts them to lowercase.
        cleaned_caption = re.sub(r'[^a-zA-Z \t\n\r\f\v]', '', caption).lower().strip()
        cleaned_caption = " ".join(cleaned_caption.split())

        image_ids.append(image_id[:-6])
        cleaned_captions.append(cleaned_caption)

    return image_ids, cleaned_captions


def build_vocab(cleaned_captions):
    """ 
    Parses training set token file captions and builds a Vocabulary object
    Args:
        cleaned_captions (str list): cleaned list of human captions to build vocab with

    Returns:
        vocab (Vocabulary): Vocabulary object
    """

    # QUESTION 1.1
    # Splits captions into words and flattens the output to return one list containing all words, including duplicates.
    corpus = list(itertools.chain(*[caption.split() for caption in cleaned_captions]))

    # Removes all words that occur 3 times or less and creates a set of the remaining words.
    counter = Counter(corpus)
    cleaned_corpus = set([word for word in corpus if counter[word] > 3])
    cleaned_corpus = list(cleaned_corpus)
    cleaned_corpus.sort()

    # create a vocab instance
    vocab = Vocabulary()

    # add the token words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the rest of the words from the cleaned captions.
    for word in cleaned_corpus:
        vocab.add_word(word)

    return vocab


def decode_caption(sampled_ids, vocab):
    """ 
    Args:
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """
    predicted_caption = ""
    for idx in sampled_ids:
        if idx == 2:
            break
        if len(predicted_caption) > 0:
            predicted_caption += ' '
        if idx > 2:
            predicted_caption += vocab.idx2word.get(idx)

    return predicted_caption


"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""
def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def print_captions(image_ids, references, candidates, index, vocab):
    print(f"Image ID: {image_ids[index]}")
    print('Reference Captions:')
    for i in range(index * 5, index * 5 + 5):
        print(references[i])
    print('Generated Caption:')
    print(decode_caption(candidates[index].tolist(), vocab))
    print()


def embed_caption(caption, vocab, decoder, device):
    caption = caption.split()
    word_embeddings = []
    for word in caption:
        if word in vocab.word2idx:
            word_embedding = decoder.embed(
                torch.tensor(vocab.word2idx[word], device=device)).cpu().detach().clone().numpy()
            word_embeddings.append(word_embedding)

    return np.mean(word_embeddings, axis=0).reshape(1, -1)
