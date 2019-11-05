#!/usr/bin/env python
# coding: utf-8

# In[38]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm


# In[56]:


def ner(sent):
    strng = nltk.word_tokenize(sent)
    strng= nltk.pos_tag(strng)
    nlp = en_core_web_sm.load()
    doc = nlp(sent)
    ans = []
    allowed = ['NNP']
    for i in range(len(strng)):
        if strng[i][1] in allowed:
            ans.append(strng[i][0])
    final =  [X.text for X in doc.ents]
    check = []
    for i in range(len(final)):
        if not final[i].isalpha():
            check.append(i)
            
    #final = list(set(final))
    if(len(final)<4):
        return final
    return final[:4]

