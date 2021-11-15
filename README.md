# Hw3_Word2Vec
###### tags: `bir` `homework`
## Enviroment
* Python 3.7.11
* Pytorch 1.10.0
* Flask 2.0.2
* NLTK 3.6.5

## Overview
* Implement **word2vec** for a set of text documents from PubMed.
* The size of text document sets could range from **1000 to 10000**.
* Preprocess the text set from document collection.
    * **Continuous Bag of Word (CBOW)**: use a window of word to predict the middle word
    * **Skip-gram (SG)**: use a word to predict the surrounding ones in window.
    * Window size is not limited.
## Data preprocessing
`Data Size: 2000 Pubmed Articles`
* 統一轉小寫，避免相同字被視為不同的字，如Covid和covid。
* 移除標點符號
* 切分成單一詞
* 去除純數字(如年分、年紀等)
* 去除停用字
```python=
# 轉小寫
fulltext = fulltext.lower()
# 去除標點符號
sentences = sent_tokenize(fulltext)
sentences = [re.sub(r'[^a-z0-9|^-]', ' ', sent.lower()) for sent in sentences]
# 切字
fulltext_split = fulltext.split()
# 去除純數字
words = [word for word in fulltext_split if not word.replace('-', '').isnumeric()]
# 移除停用字
stop = stopwords.words('english')
cleanWords = [word for word in words if word not in stop]
```
* Lemmatization
```python=
lemmatizer = WordNetLemmatizer()
tag_word = nltk.pos_tag(cleanWords)
lemmaWords = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag[1])) for word, tag in zip(cleanWords, tag_word)]
```
## Skip-gram
![](https://i.imgur.com/ERf66Ut.png)
* 輸入為一個詞W1, 輸出為Wo1,...,Woc,c = windows size
* e.g. “I drive my car to the store”
    ```
    input: 'car ' 
    output: '{'I', 'drive', 'my', 'to', 'the', 'store'}'
    ```
* Steps:
    1. 選定句子中間的一個詞作為input word
    2. 定義skip_window(左右各取多少字)
    3. 定義num_skips(從window中取多少不同的詞作為output word)
![](https://i.imgur.com/3jjAYEX.jpg)
* 需先對data作one-hot encoding
![](https://i.imgur.com/wFKaBek.jpg)
* 根據window size建立context pair
```python=
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
```
* 建立詞彙表
* 根據詞彙表將context pair轉為數字
```python=
word2id = {w:i for i,w in enumerate(vocab)}
id2word = {i:w for i,w in enumerate(vocab)}

data = {}
data['x'] = torch.tensor([word2id[w[0]] for w in context_pair])
data['y'] = torch.tensor([word2id[w[1]] for w in context_pair])
```
* Model
``` python=
class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, 1)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        out = self.embeddings(x)
        out = self.linear(out)
        return out
```
## Result
* Input Word: `covid-19`
* Output: 
    * Similar Words

        `patient` : 11.63%

        `diagnosis` : 7.46%

        `vaccine` : 3.61%

        `pandemic` : 3.43%

        `disease` : 2.79%

        `primerdesign` : 2.39%

        `coronavirus` : 2.08%

        `infection` : 1.95%

        `vaccination` : 1.40%

        `test` : 1.33%
    * Visualize

        ![](https://i.imgur.com/ZoqZNgC.png)
## Reference
* [Python 文字資料處理](https://www.itread01.com/content/1548704733.html)
* [[自然語言處理] #2 Word to Vector 實作教學 (實作篇)](https://medium.com/royes-researchcraft/%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86-2-word-to-vector-%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-%E5%AF%A6%E4%BD%9C%E7%AF%87-e2c1be2346fc)
* [Word2Vec Implementation](https://towardsdatascience.com/a-word2vec-implementation-using-numpy-and-python-d256cf0e5f28)
* [Word2vec from scratch (Skip-gram & CBOW)](https://medium.com/@pocheng0118/word2vec-from-scratch-skip-gram-cbow-98fd17385945)
* [Skip-Gram: NLP context words prediction algorithm](https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c)
* [CBOW preprocessing](https://www.kaggle.com/jarfo1/cbow-preprocessing)
* [ Implementing Deep Learning Methods and Feature Engineering for Text Data](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)
* [An implementation guide to Word2Vec using NumPy and Google Sheets](https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281)
* [詞向量 Word Embedding](https://blog.maxkit.com.tw/2020/08/word-embedding.html)
* [讓電腦聽懂人話: 直觀理解 Word2Vec 模型](https://medium.com/@tengyuanchang/%E8%AE%93%E9%9B%BB%E8%85%A6%E8%81%BD%E6%87%82%E4%BA%BA%E8%A9%B1-%E7%90%86%E8%A7%A3-nlp-%E9%87%8D%E8%A6%81%E6%8A%80%E8%A1%93-word2vec-%E7%9A%84-skip-gram-%E6%A8%A1%E5%9E%8B-73d0239ad698)