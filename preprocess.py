import pickle
import os
import string
from typing import Counter
import nltk
from nltk.corpus.reader import tagged
from nltk.util import pr
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

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


if __name__=='__main__':
    # load data
    docPath  = f'{os.getcwd()}/PubmedDoc.pkl'
    numDoc = 1000
    with open(docPath, 'rb') as fpick:
        pmDoc = pickle.load(fpick)
    fulltext = []
    for i, (title, article) in enumerate(pmDoc.items()):
        # if title:
        #     fulltext.append(title)
        # for text in article.values():
        #     if text:
        #         fulltext.append(text)
        if i <= numDoc:
            if title:
                fulltext.append(title)
            for text in article.values():
                if text:
                    fulltext.append(text)
        else:
            break

    # 清理
    fulltext = ' '.join(fulltext)
    sentences = sent_tokenize(fulltext)
    sentences = [re.sub(r'[^a-z0-9|^-]', ' ', sent.lower()) for sent in sentences]
    stop = stopwords.words('english')
    cleantext = []
    for sent in tqdm(sentences):
        words = [word for word in sent.split() if not word.replace('-', '').isnumeric()]
        cleanwords = [word for word in words if word not in stop]
        lemmatizer = WordNetLemmatizer()
        tag_word = nltk.pos_tag(cleanwords)
        lemmaWords = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag[1])) for word, tag in zip(cleanwords, tag_word)]
        cleantext.append(' '.join(lemmaWords))

    with open(f'train_data_{numDoc}_lemma.pkl', 'wb') as fpick:
        pickle.dump(cleantext, fpick)

    # # preprocess
    # windowSize = 2
    # context_pair = []
    # vocab  = set()
    # for sent in cleantext:
    #     context_pair += get_context_pair(sent, windowSize)
    #     word_set = get_word_set(sent)
    #     for word in word_set:
    #         vocab.add(word)

    # wordSize = len(vocab)

    # word2id = {w:i for i,w in enumerate(vocab)}
    # id2word = {i:w for i,w in enumerate(vocab)}

    # data_pair = [(word2id[w[0]], word2id[w[1]]) for w in context_pair]


