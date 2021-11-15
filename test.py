from torch.utils.data.dataloader import DataLoader
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import json, pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def display_pca_scatterplot(model, words, vocab):
    # Take word vectors
    word_vectors = []
    for word in words:
        x = torch.tensor(vocab[word])
        out = model(x.cuda())
        out = F.softmax(out)
        out = out.detach().cpu().numpy()
        word_vectors.append(out)
    word_vectors = np.array(word_vectors)
    # word_vectors = np.array([model[w] for w in words])

    # PCA, take the first 2 principal components
    twodim = PCA().fit_transform(word_vectors)[:,:2]

    # Draw
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x, y, word)
    plt.show()

if __name__=="__main__":
    search_word = "covid-19"
    dirpath = 'ckpy/train_data_2000_lemma_128'
    with open(f'{dirpath}/word2id.json', 'r')as fjson:
        vocab = json.load(fjson)

    with open(f'{dirpath}/id2word.json', 'r')as fjson:
        ids = json.load(fjson)

    model = torch.load(f'{dirpath}/best.pth')
    if search_word not in vocab.keys():
        print(search_word, ' not in vocab')
    else:
        x = torch.tensor(vocab[search_word])
        out = model(x.cuda())
        out = F.softmax(out)
        out = out.detach().cpu().numpy()
    
    result = {ids[str(i)]:(p*100) for i,p in enumerate(out)}
    result = sorted(result.items(), key=lambda x:x[1], reverse=True)
    words = [word[0] for word in result[:10]]
    words.append(search_word)
    display_pca_scatterplot(model, words, vocab)
    for word, prob in result[:10]:
        print(word, ':', round(prob, 3))