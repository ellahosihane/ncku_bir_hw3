from unicodedata import lookup
from torch.serialization import save
from torch.utils.data.dataloader import DataLoader
import processor
import torch
import word2vec_model
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import json, pickle
import os

if __name__=='__main__':
    filename = 'train_data_2000_lemma'
    batch_size = 128
    epochs = 1000
    windowSize = 2
    best_loss = 1000
    lr = 0.001
    save_path = f'ckpt/{filename}_{batch_size}'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Get data
    word2id, id2word, vocab_size, data = processor.get_data(f'{filename}.pkl', windowSize)

    ratio = .2
    fulldata = TensorDataset(data['x'], data['y'])
    test_size = int(ratio*len(fulldata))
    train_size = len(fulldata)-test_size
    train_data, test_data = torch.utils.data.random_split(fulldata, [train_size, test_size])
    train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Get Model
    model = word2vec_model.SkipGram_Model(vocab_size, 300).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #training
    lossList = {"train":[], "test":[]}
    for epoch in range(epochs):
        #train
        model.train()
        losses = []
        for i, (x, y) in enumerate(tqdm(train)):
            x = x.cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.cuda())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = np.mean(losses)
        lossList["train"].append(epoch_loss)
        #eval
        model.eval()
        losses = []
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(test)):
                x = x.cuda()
                output = model(x)
                loss = criterion(output, y.cuda())
                losses.append(loss.item())
            epoch_loss = np.mean(losses)
            lossList["test"].append(epoch_loss)
        
        print("Epoch = {:3}/{}, train_loss = {:10}, test_loss = {:10}".format(epoch+1, epochs, round(lossList["train"][epoch], 7), round(lossList["test"][epoch], 7)))
        #save
        if epoch_loss < best_loss:
            torch.save(model, f'{save_path}/best.pth')
            best_loss = epoch_loss
        else:
            torch.save(model, f'{save_path}/last.pth')
    
    print('Done!')
    with open(f'{save_path}/Loss.json', 'w')as fjson:
        json.dump(lossList, fjson)
    with open(f'{save_path}/id2word.json', 'w')as fjson:
        json.dump(id2word, fjson)
    with open(f'{save_path}/word2id.json', 'w')as fjson:
        json.dump(word2id, fjson)
        