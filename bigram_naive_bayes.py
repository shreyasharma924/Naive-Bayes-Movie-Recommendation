# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=1.0, bigram_laplace=0.005, bigram_lambda=0.5, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    
    pos_cnt_uni = Counter()
    neg_cnt_uni = Counter()
    pos_cnt_bi = Counter()
    neg_cnt_bi = Counter()
    posWords_uni = 0
    negWords_uni = 0
    posWords_bi = 0
    negWords_bi = 0
    
    for review, label in zip(train_set, train_labels):
        if label == 1:
            for word in review:
                pos_cnt_uni.update([word.lower()])
                posWords_uni += 1
        elif label == 0:
            for word in review:
                neg_cnt_uni.update([word.lower()])
                negWords_uni += 1 

    for review, label in zip(train_set, train_labels):
        if label == 1:
            for c in range (len(review)-1):
                word = review[c]+ ' ' + review[c+1]
                pos_cnt_bi.update([word.lower()])
                posWords_bi += 1
        elif label == 0:
            for c in range (len(review)-1):
                word = review[c]+ ' ' + review[c+1]
                neg_cnt_bi.update([word.lower()])
                negWords_bi += 1 

    yhats = []

    unique_wrds_uni = len(pos_cnt_uni) + len(neg_cnt_uni)
    unique_wrds_bi = len(pos_cnt_bi) + len(neg_cnt_bi)

    neg_prior = 1- pos_prior
    for doc in tqdm(dev_set, disable=silently):
        neg = 0 # for the unigram 
        pos = 0 # for the unigram 
        neg_bi = 0 
        pos_bi = 0
        neg_final = 0 
        pos_final = 0 

        for word_ in doc:
            word = word_.lower()
            pos += math.log10((pos_cnt_uni[word] + unigram_laplace)/(posWords_uni + unigram_laplace * unique_wrds_uni))
            neg += math.log10((neg_cnt_uni[word] + unigram_laplace)/(negWords_uni + unigram_laplace * unique_wrds_uni))

        for c in range(len(doc)-1):
            words = doc[c]+ ' '+ doc[c+1]
            pos_bi += math.log10((pos_cnt_bi[words.lower()] + bigram_laplace)/(posWords_bi + bigram_laplace * unique_wrds_bi))
            neg_bi += math.log10((neg_cnt_bi[words.lower()] + bigram_laplace)/(negWords_bi + bigram_laplace * unique_wrds_bi))

        
        pos = math.log10(pos_prior)+ pos
        neg = math.log10(neg_prior)+ neg 
        pos_bi = math.log10(pos_prior)+ pos_bi
        neg_bi = math.log10(neg_prior)+ neg_bi
        
        neg_final= ((1-bigram_lambda)* neg) + (bigram_lambda * neg_bi)
        pos_final= ((1-bigram_lambda)* pos) + (bigram_lambda * pos_bi)   

        if pos_final >= neg_final:
            x = 1
        else:
            x = 0

        yhats.append(x)
    return yhats
