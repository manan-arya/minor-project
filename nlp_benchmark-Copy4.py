#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
from sklearn.model_selection import StratifiedShuffleSplit
from WordEmbeding.word_embeding_classifier import WordEmebdingClassifier
#from InferSent.classifier import InferSentClassifier,XGBoostClassifier
from TFIDFVector.tfidf_classifier import TFIDFClassifer
import text_summarization_using_spacy as summarizer
from SemHash.semhash_classifier import SemHashClassfier
import pandas as pd
import matplotlib.pyplot as plt
import re
import NER


# In[2]:


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = [re.sub(r"[^a-zA-Z.]+", ' ', k) for k in cleantext.split("\n")]
    return cleantext


# In[3]:


nlu_data ='AskUbuntuCorpus.json'


# In[4]:


temp_data = None
with open(nlu_data,'r') as f:
    temp_data = json.load(f)


# In[ ]:





# In[5]:


texts = []
labels = []

for s in temp_data['sentences']:
    try:
        texts.append(s['text'].lower() + s['answer']['text'].lower())
    except:
        texts.append(s['text'].lower())
    labels.append(s['intent'])
    


# In[6]:


for i in range(len(texts)):
    texts[i] = cleanhtml(texts[i])
    texts[i] = ' '.join(texts[i])


# In[7]:


sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2) 


# In[8]:


train_index, text_index = list(sss.split(texts,labels))[0]


# In[9]:


train_texts = [texts[index] for index in train_index]
train_labels = [labels[index] for index in train_index]
test_texts = [texts[index] for index in text_index]
test_labels = [labels[index] for index in text_index]


# In[10]:


for i in range(len(train_texts)):
    check =  summarizer.summarize(train_texts[i])
    if check and len(check)>10:
        train_texts[i] = check


# In[11]:


import NER
entities = []
for i in range(len(test_texts)):
    entities.append(NER.ner(test_texts[i]))


# ## Compare

# In[12]:


classifiers =["SemHashClassfier"]


# In[13]:


result = []
for cls_name in classifiers:
    cls = globals()[cls_name]()
    model,le = cls.train(train_texts,train_labels)
    acc = cls.eval(test_texts,test_labels)
    #print(cls.predict(test_texts))
    result.append([cls_name,acc])


# In[14]:


test_texts = cls.doc2vec(test_texts)


# In[15]:


ans = model.predict(test_texts)
le.inverse_transform(ans)


# In[16]:


temp_test = cls.doc2vec(['After upgrade Ubuntu 16.04 my system display showing only Ubuntu splash logo, can not boot device, please help me for this.'])
temp = model.predict(temp_test)
le.inverse_transform(temp)


# In[17]:


type(model)


# In[18]:


df = pd.DataFrame(result, columns = ['Name', 'Accuracy']) 


# In[19]:


df


# In[20]:


fig, ax = plt.subplots()
ax.barh(df['Name'], df['Accuracy'], align='center')
ax.set_yticklabels(df['Name'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('Intent Rec Accuracy')


# ## Conclusion

# We compare several different sentence embeding for intent classification.
# 1. Simple sum of wordembeding, 
# 2. Facebook InferSent.
# 3. TF-IDF
# 4. Semhash, https://arxiv.org/abs/1810.07150
# 
# ### Simple sum the word embeding achieve the best accuray in intention classification.
# This is quiet interesting. Why could simple sum of wording embeding represent the sentence very well?

# In[21]:


i = 0
inp = "What IDEs are available for Ubuntu?"
entities = NER.ner(inp)
test_texts = cls.doc2vec(inp)
ans = model.predict(test_texts)
le.inverse_transform(ans)
print("Intent:"+str(le.inverse_transform([ans[i]])[0]))
for i in range(len(entities)):
    if entities[i].isalnum():
        print("Entity:"+entities[i])

