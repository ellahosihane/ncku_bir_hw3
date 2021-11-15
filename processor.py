import pickle, json
import numpy as np
import os, re, random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


def get_context_pair(sentence, window_size):
    pair = []
    sent_split  = sentence.split()
    for i, word in enumerate(sent_split):
        start = i - window_size if i - window_size >= 0 else 0
        end = i + window_size if i + window_size <= len(sent_split) else len(sent_split)
        for j in range(start, end):
            if j != i:
                pair.append((word, sent_split[j]))
    return pair

def get_word_set(sentence):
    wordSet = set()
    for word in sentence.split():
        wordSet.add(word)
    return wordSet

def get_data(file, window_size):
    with open(file, 'rb') as fpick:
        text = pickle.load(fpick)
    context_pair = []
    vocab  = set()
    for sent in text:
        context_pair += get_context_pair(sent, window_size)
        word_set = get_word_set(sent)
        for word in word_set:
            vocab.add(word)

    wordSize = len(vocab)

    word2id = {w:i for i,w in enumerate(vocab)}
    id2word = {i:w for i,w in enumerate(vocab)}

    data = {}
    data['x'] = torch.tensor([word2id[w[0]] for w in context_pair])
    data['y'] = torch.tensor([word2id[w[1]] for w in context_pair])

    return word2id, id2word, wordSize, data

if __name__ == '__main__':
    word2id, id2word, wordSize, data = get_data('train_data_1000.pkl', 2)
    train = DataLoader(dataset=TensorDataset(data["x"], data["y"]),
                            batch_size=32, shuffle=False)
    for i,(data,label) in enumerate(train):
        print(data,label)