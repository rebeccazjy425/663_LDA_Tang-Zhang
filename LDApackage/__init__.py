import numpy as np
import string
import random
from collections import OrderedDict

topwordsfile = 'topwords.dat'
stopword = 'stopword.txt'

class documenting(object):
    """Auxillary class to store preprocessed text"""
    def __init__(self):
        self.words = []
        self.length = 0

class dataprocessing(object):
    """Predefine ordered dictionary to store data"""
    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()
                
def preprocessing(datafile):
    """Preprocessing the text file"""
    with open(datafile, 'r') as f:
        docs = f.readlines()
    with open(stopword, 'r') as s:
        stopwords = s.readlines()
    stpw = []
    for word in stopwords:
        stpw.append(word.strip())
    pred = dataprocessing()
    items_idx = 0
    for line in docs:
        if line != "":
            tmp = []
            before = line.strip().split()
            #Remove stopwords from the strings
            for word in before:
                if word.lstrip(string.punctuation).rstrip(string.punctuation).lower() not in stpw:
                    if word != "":
                        tmp.append(word.lstrip(string.punctuation).rstrip(string.punctuation).strip())
            doc = documenting()
            for item in tmp:
                if item in pred.word2id:
                    doc.words.append(pred.word2id[item])
                else:
                    pred.word2id[item] = items_idx
                    doc.words.append(items_idx)
                    items_idx += 1
            doc.length = len(tmp)
            pred.docs.append(doc)
        else:
            pass
    pred.docs_count = len(pred.docs)
    pred.words_count = len(pred.word2id)
    return pred

class LDAModel(object):
    
    def __init__(self,pred):
        """Initialize parameters, matrices and files"""
        self.pred = pred 
        self.topwordsfile = topwordsfile
        #Prior parameter definitions
        self.K = 5
        self.beta = 0.1
        self.alpha = 0.1
        self.number_iter_times = 100
        self.top_words_num = 10
        
        #Initialize empty matrices to store values
        self.p = np.zeros(self.K)   
        self.nword = np.zeros((self.pred.words_count,self.K),dtype="int")       
        self.nwordsum = np.zeros(self.K,dtype="int")    
        self.ndoc = np.zeros((self.pred.docs_count,self.K),dtype="int")       
        self.ndocsum = np.zeros(pred.docs_count,dtype="int")    
        self.Z = np.array([[0 for y in range(pred.docs[x].length)] for x in range(pred.docs_count)])

        for x in range(len(self.Z)):
            self.ndocsum[x] = self.pred.docs[x].length
            for y in range(self.pred.docs[x].length):
                topic = random.randint(0,self.K-1)
                self.Z[x][y] = topic
                self.nword[self.pred.docs[x].words[y]][topic] += 1
                self.ndoc[x][topic] += 1
                self.nwordsum[topic] += 1

        self.theta = np.zeros((self.pred.docs_count, self.K))
        self.phi = np.zeros((self.K, self.pred.words_count))

    def sampling(self,i,j):
        """Gibbs sampling algorithm"""
        topic = self.Z[i][j]
        word = self.pred.docs[i].words[j]
        self.nword[word][topic] -= 1
        self.ndoc[i][topic] -= 1
        self.nwordsum[topic] -= 1
        self.ndocsum[i] -= 1

        Beta = self.pred.words_count * self.beta
        Kappa = self.K * self.alpha
        self.p = (self.nword[word] + self.beta)/(self.nwordsum + Beta)*(self.ndoc[i] + self.alpha) / (self.ndocsum[i] + Kappa)
        for k in range(1,self.K):
            self.p[k] += self.p[k-1]

        u = random.uniform(0,self.p[self.K-1])
        for topic in range(self.K):
            if self.p[topic]>u:
                break
        self.nword[word][topic] +=1
        self.nwordsum[topic] +=1
        self.ndoc[i][topic] +=1
        self.ndocsum[i] +=1
        return topic
    
    def estimate(self):
        """Estimation"""
        for x in range(self.number_iter_times):
            for i in range(self.pred.docs_count):
                for j in range(self.pred.docs[i].length):
                    topic = self.sampling(i,j)
                    self.Z[i][j] = topic
        self._theta()
        self._phi()
        self.save()
    
    def _theta(self):
        """Updating theta value"""
        for i in range(self.pred.docs_count):
            self.theta[i] = (self.ndoc[i]+self.alpha)/(self.ndocsum[i]+self.K * self.alpha)
    def _phi(self):
        """Updating phi value"""
        for i in range(self.K):
            self.phi[i] = (self.nword.T[i] + self.beta)/(self.nwordsum[i]+self.pred.words_count * self.beta)
    def save(self):
        """Save top words to file"""
        with open(self.topwordsfile,'w') as f:
            self.top_words_num = min(self.top_words_num,self.pred.words_count)
            for x in range(self.K):
                f.write('Topic ' + str(x) + '\n')
                twords = []
                twords = [(n,self.phi[x][n]) for n in range(self.pred.words_count)]
                twords.sort(key = lambda i:i[1], reverse= True)
                for y in range(self.top_words_num):
                    word = OrderedDict({value:key for key, value in self.pred.word2id.items()})[twords[y][0]]
                    f.write('\t'*2+ word +'\t' + str(twords[y][1])+ '\n')

                    
###################### below is EM ###########################
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:54:50 2019

@author: Xinghong Tang
"""

import numpy as np
import codecs
import jieba
import re
import random
from scipy.special import gammaln   # log(gamma(x))
from scipy.special import psi        #digamma fcn
from scipy.special import polygamma  #derivative of psi

import gensim
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
np.random.seed(2018)
import nltk
nltk.download('wordnet')



# itemIdList : the list of distinct terms in the document
# itemCountList : the list of number of the existence of corresponding terms
# wordCount : the number of total words (not terms)
class Document:
    def __init__(self, itemIdList, itemCountList, wordCount):
        self.itemIdList = itemIdList
        self.itemCountList = itemCountList
        self.wordCount = wordCount



def lemmatize_and_stem(text):
    """Process lemmatizing and stemming for a single word"""
    lmtzr = WordNetLemmatizer()
    stemmer = PorterStemmer()
    return stemmer.stem(lmtzr.lemmatize(text, pos = 'v'))


def read_documents(filepath):
    """read documents"""
    with open(filepath, 'r') as f:
        documents = f.read().split('</TEXT>')
    return documents

def read_documents_space(filepath):
    """read documents seperated by space"""
    with open(filepath, 'r') as f:
        documents = f.read().split('\n')
    return documents

# preprocessing (segmentation, stopwords filtering, represent documents as objects of class Document)
def empreprocessing(documents):
    """preprocess documents
    simple segmentation, lemmatizing and stemming of words, checking stopwords, chekcing length of words    
    rtype: docs: class of document
           word2index, index2word: conversion between words and their index
    """
    # read the list of stopwords from stopword.txt
    with open('stopword.txt','r') as s:
        stopwords = s.read().split('\n')
    
    docs = []
    word2index = {}
    index2word = {}
    
    currentWordId = 0
    for document in documents:
        word2Count = {}
        # segmentation
        wordlist = jieba.cut(document)
        for word in wordlist: 
            word = word.lower().strip()
            # filter the stopwords
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
                word = lemmatize_and_stem(word)
                
                #get a list of unique words
                if word not in word2index:
                    word2index[word] = currentWordId
                    index2word[currentWordId] = word
                    currentWordId += 1
                if word in word2Count:
                    word2Count[word] += 1
                else:
                    word2Count[word] = 1
        itemIdList = []
        itemCountList = []
        wordCount = 0

        for word in word2Count.keys():
            itemIdList.append(word2index[word])
            itemCountList.append(word2Count[word])
            wordCount += word2Count[word]

        docs.append(Document(itemIdList, itemCountList, wordCount))

    return docs, word2index, index2word
    
def MaxWordNum(M,docs):
    num = 0
    for d in range(M):
        if len(docs[d].itemIdList) > num:
            num = len(docs[d].itemIdList)
    return num



def update_beta(topic_word,log_beta):
    """
    M-step for updating beta
    type: nzw: (K,V): # topic-word count
    rtype: log_beta : (K,V)
    """
    K,V = np.shape(topic_word)
    topic_word_totals = np.sum(topic_word,axis =1) #(k,)
    
    for k in range(K):
        for w in range(V):
            if(topic_word[k, w] > 0):
                log_beta[k, w] = np.log(topic_word[k, w]) - np.log(topic_word_totals[k])
            else:
                # avoid negative infinities
                log_beta[k, w] = -100
    
    return log_beta



def update_alpha(gamma,M,K,threshold=1e-5, max_iter=100):
    """
    M step for updating alpha (hyperparameter of theta) using Netwon-Ralphson method
    gamma (M,K): hidden parameter for theta
    rtype: alpha 
    """
    init_alpha = 100
    log_alpha = np.log(init_alpha)

    digamma_gamma = psi(gamma)
    digamma_sum_gamma = psi(np.sum(gamma, axis=1))
    Alphass = np.sum(digamma_gamma - digamma_sum_gamma.reshape((M, 1))) 
    
    i = 0
    while i in range(max_iter):
        alpha = np.exp(log_alpha)
        if np.isnan(alpha):
            init_alpha *=10
            alpha = init_alpha
            log_alpha = np.log(alpha)
        
        L_alpha = compute_L_alpha(M,K,alpha,Alphass)
        d1L_alpha = compute_d1L_alpha(M,K,alpha,Alphass)
        d2L_alpha = compute_d2L_alpha(M,K,alpha)
        log_alpha = log_alpha -d1L_alpha /(d2L_alpha * alpha +d1L_alpha)
        i +=1
        
        if np.abs(d1L_alpha) > threshold:
            break
     
    return np.exp(log_alpha)


def compute_L_alpha(M,K,alpha,Alphass):
    """extract ELBO terms containing alpha"""
    return M * (gammaln(K * alpha) - K * gammaln(alpha)) + (alpha - 1) * Alphass
    
def compute_d1L_alpha(M,K,alpha,Alphass):
    """first derivative of ELBO of alpha"""
    return M * (K * psi(K * alpha) - K* psi(alpha)) + Alphass
    
def compute_d2L_alpha(M,K,alpha):
    """second derivative of ELBO of alpha"""
    return M * (K**2 * polygamma(1,K*alpha) - K *polygamma(1,alpha))



def update_parameters(docs, d, gamma, phi,K,alpha,log_beta,max_iteration = 20):
    """
    update gamma and phi for each document
    d,K: index for current document, total topic numbers
    gamma, phi: parameters from last update for previous document
    alpha, beta: updated model parameters from last M-step
    max_iteration = 20: set default number of iteration
    """
    phisum = 0
    oldphi = np.zeros([K])
    digamma_gamma = np.zeros([K]) #digamma_gamma per document 1d-array
    
    #initialize gamma values to alpha + (number of words in document)/K
    gamma[d,:] = np.full((K),alpha) + docs[d].wordCount * np.ones(K) / float(K)
    digamma_gamma = psi(gamma[d,:])
    
    #initialize phis to 1/K; only need to consider each word type
    phi = np.ones((len(docs[d].itemIdList),K)) / float(K)

    #update phi and gamma
    for iteration in range(max_iteration):
        for w in range(len(docs[d].itemIdList)):
            phisum = 0
            for k in range(K):
                oldphi[k] = phi[w, k]
                phi[w, k] = digamma_gamma[k] + log_beta[k, docs[d].itemIdList[w]]
                if k > 0:
                    phisum = np.log(np.exp(phisum) + np.exp(phi[w, k]))
                else:
                    phisum = phi[w, k]
                    
            for k in range(K):
                phi[w, k] = np.exp(phi[w, k] - phisum)
                gamma[d,k] =  gamma[d,k] + docs[d].itemCountList[w] * (phi[w, k] - oldphi[k])
                digamma_gamma[k] = psi(gamma[d,k])

    return phi,gamma



def initialLdaModel(K,V,topic_word,log_beta):
    """initialize LDA model
    set an arbitrary topic-word distribution
    """
    topic_word = np.ones((K,V))/V + np.random.rand(K,V)
    log_beta = update_beta(topic_word,log_beta)
    
    return log_beta



def LDA_VIEM(documents,num_topic,maxTopicWordsNum_show, iterInference = 20, iterEM = 20):
    """get parameters alpha and beta using variation inference and EM algorithm
    
    documents: pass as argument in function preprocessing(documents)
    num_topic: user can define how many topics they want to get
    maxTopicWordsNum_show: user can select how many top words in each topic
    iterInference = 20 (default): iteration times of variational inference w/o checking convergence
    iterEM = 20 (default): iteration times of variational EM algorithm w/o checking convergence
    
    rtype: alpha, log_beta: model parameters
           topicwords: 'num_topic' topics and their top 'maxTopicWordsNum_show' words
    """
    
    #load data
    docs, word2index, index2word = empreprocessing(documents) 
    print('Word preprocessing is finished.')
    
    # number of documents
    M = len(docs)
    # number of total distinct terms for all documents
    V = len(word2index)
    # number of topic
    K = num_topic

    
    #inital alpha
    alpha = 5
    # sufficient statistic of alpha
    alphaSS = 0
    # the topic-word distribution 
    log_beta = np.zeros([K, V])
    # topic-word count, this is a sufficient statistic to calculate beta
    topic_word = np.zeros([K, V])
    
    # inference parameter gamma
    gamma = np.zeros([M, K])
    # inference parameter phi
    phi = np.zeros([MaxWordNum(M,docs), K])
    
    # initialization of the model parameter beta, the update of alpha is ommited
    log_beta = initialLdaModel(K,V,topic_word,log_beta)
    
    # EM Algorithm
    for iteration in range(iterEM): 
        print('{}th EM iteration is in process.'.format(iteration+1))
        topic_word_totals = np.zeros([K])
        topic_word = np.zeros([K, V])
        alphaSS = 0
        
        # E Step
        for d in range(M):
            phi,gamma= update_parameters(docs, d, gamma, phi,K,alpha,log_beta)
            gammaSum = 0
            for k in range(K):
                gammaSum += gamma[d, k]
                alphaSS += psi(gamma[d, k])
            alphaSS -= K * psi(gammaSum)
    
            for w in range(len(docs[d].itemIdList)):
                for k in range(K):
                    topic_word[k][docs[d].itemIdList[w]] += docs[d].itemCountList[w] * phi[w, k]
                    topic_word_totals[k] += docs[d].itemCountList[w] * phi[w, k]
    
        # M Step
        # update global parameters: beta and alpha
        log_beta = update_beta(topic_word,log_beta)
        alpha = update_alpha(gamma,M,K) #scalar alpha

    # calculate the top 10 terms of each topic
    topicwords = []
    maxTopicWordsNum =  maxTopicWordsNum_show
    for k in range(0, K):
        ids = log_beta[k, :].argsort()
        topicword = []
        for j in ids:
            topicword.insert(0, index2word[j])
        topicwords.append(topicword[0 : min(maxTopicWordsNum, len(topicword))])
    
    return alpha, log_beta,topicwords