from __future__ import unicode_literals, print_function, division
from io import open
import torch
import re
import  numpy as np
import gensim
from torch.utils.data import Dataset
from Sentiment_Analysis_Config import Config

class Data_set(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        if Label is not None:#考虑对测试集的使用
            self.Label = Label
    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        if self.Label is not None:
            data = torch.from_numpy(self.Data[index])
            label = torch.from_numpy(self.Label[index])
            return data, label
        else:
            data = torch.from_numpy(self.Data[index])
            return data


def stopwordslist():#创建停用词表
    stopwords = [line.strip() for line in open(Config.stopword_path, encoding='UTF-8').readlines()]
    return stopwords


def build_word2id(file):
    ''' 
    @description: 
    @param {*}file: word2id保存地址
    @return {*}None
    '''     
    
    #num_50=0#统计长度大于50的句子数
    stopwords = stopwordslist()
    word2id = {'_PAD_': 0}
    path = [Config.train_path, Config.val_path]
    #print(path)

    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                out_list = []
                # 去停用词
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in stopwords:
                        rt = re.findall('[a-zA-Z]+', word)
                        if word != '\t':
                            # if is_number(word):
                            # continue
                            if len(rt) == 1:
                                continue
                            else:
                                out_list.append(word)
                for word in out_list:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    with open(file, 'w', encoding='utf-8') as f:
        for w in word2id:
            f.write(w+'\t')
            f.write(str(word2id[w]))
            f.write('\n')

def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """

    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs


def text_to_array(word2id, seq_lenth, path):  # 文本转为索引数字模式,

    lable_array=[]
    i=0
    sa=[]
    #获取句子个数
    with open(path, encoding='utf-8') as f1:
        for l1 in f1.readlines():
            s= l1.strip().split()
            s1=s[1:]
            new_s = [word2id.get(word, 0) for word in s1]  # 单词转索引数字
            sa.append(new_s)
        #print(len(sa))

    with open(path, encoding='utf-8') as f:
        sentences_array=np.zeros(shape=(len(sa),seq_lenth))#行：句子个数 列：句子长度
        for line in f.readlines():
            sl1 = line.strip().split()
            sen=sl1[1:]
            new_sen = [word2id.get(word, 0) for word in sen]  # 单词转索引数字,不存在则为0
            new_sen_np=np.array(new_sen).reshape(1,-1)

            #补齐每个句子长度，多余补零，少了就直接赋值,0填在前面。
            if np.size(new_sen_np,1)<seq_lenth:
                sentences_array[i,seq_lenth-np.size(new_sen_np,1):]=new_sen_np[0,:]
            else:
                sentences_array[i, 0:seq_lenth]=new_sen_np[0,0:seq_lenth]

            i=i+1
            lable=int(sl1[0])#标签
            lable_array.append(lable)
    return np.array(sentences_array),lable_array

def text_to_array_nolable(word2id,seq_lenth ,path):  # 文本转为索引数字模式,

    i=0
    sa=[]
    #获取句子个数
    with open(path, encoding='utf-8') as f1:
        for l1 in f1.readlines():
            s= l1.strip().split()
            s1=s[0:] #s1=s[1:]
            new_s = [word2id.get(word, 0) for word in s1]  # 单词转索引数字
            sa.append(new_s)
    with open(path, encoding='utf-8') as f:
        sentences_array=np.zeros(shape=(len(sa),seq_lenth))#行：句子个数 列：句子长度
        for line in f.readlines():
            sl1 = line.strip().split()
            sen=sl1[0:] #sen=sl1[1:]
            new_sen = [word2id.get(word, 0) for word in sen]  # 单词转索引数字,不存在则为0
            new_sen_np=np.array(new_sen).reshape(1,-1)
            if np.size(new_sen_np,1)<seq_lenth:
                sentences_array[i,seq_lenth-np.size(new_sen_np,1):]=new_sen_np[0,:]
            else:
                sentences_array[i, 0:seq_lenth]=new_sen_np[0,0:seq_lenth]
            i=i+1
    return np.array(sentences_array)


def to_categorical(y, num_classes=None):#将类别转化为one-hot编码
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def prepare_data(w2id, train_path,val_path,test_path,seq_lenth):#得到数字索引表示的句子和标签
    train_array,train_lable = text_to_array(w2id,seq_lenth= seq_lenth,path=train_path)
    val_array,val_lable  = text_to_array(w2id,seq_lenth=seq_lenth,path= val_path)
    test_array,test_lable=text_to_array(w2id,seq_lenth=seq_lenth,path=test_path)

    #标签为[1, 1, 1, 1, 1, 1, 1, 1, 0, 0...]将标签转为onehot
    #train_lable=to_categorical(train_lable,num_classes=2)
    #val_lable=to_categorical(val_lable,num_classes=2)

    """for i in train_lable:
        np.array([i])"""
    train_lable=np.array([train_lable]).T
    val_lable=np.array([val_lable]).T
    test_lable=np.array([test_lable]).T
    """转换后标签
            [[0. 1.]
            [0. 1.]
            [0. 1.]
            ...
            [1. 0.]
            [1. 0.]
            [1. 0.]]"""
    #print(train_lab,"\nval\n",val_lab)
    return train_array ,train_lable,val_array,val_lable,test_array,test_lable



#建立word2id
build_word2id('./word2vec_data/word2id.txt')#建立word2id单词转索引数字
splist=[]
word2id={}
with open('./word2vec_data/word2id.txt', encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()#去掉\n \t 等
            splist.append(sp)
        word2id=dict(splist)#转成字典


for key in word2id:# 将字典的值，从str转成int
    word2id[key]=int(word2id[key])

id2word={}#得到id2word
for key,val in word2id.items():
    id2word[val]=key
#建立word2vec
w2vec=build_word2vec(Config.pre_word2vec_path,word2id,Config.corpus_word2vec_path)

#得到句子id表示和标签
train_array,train_lable,val_array,val_lable,test_array,test_label=prepare_data(word2id,
                                                         train_path=Config.train_path,
                                                         val_path=Config.val_path,
                                                         test_path=Config.test_path,seq_lenth=Config.max_sen_len)

np.savetxt('./word2vec/train_data.txt', train_array,fmt='%d')
np.savetxt('./word2vec/val_data.txt', val_array,fmt='%d')
np.savetxt('./word2vec/test_data.txt', test_array,fmt='%d')

