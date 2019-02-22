
# coding: utf-8

# In[21]:


### importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[22]:


# initializing NLTK and stop words

import nltk
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordNERTagger
stop = stopwords.words('english')


# In[3]:


### importing NLTK features for NER
from nltk import word_tokenize, pos_tag, ne_chunk


# In[301]:


### opening and reading the desired file
file_name = '857737_2005-03-11_EMPLOYMENT AGREEMENT - WARREN CLAMEN.txt'


doc = open("document-analytics-master/document-analytics-master/employment contracts/"+file_name, "r", encoding="utf8")
doc = doc.read()


# In[302]:


### removing new line characters and extra spaces

doc = doc.replace('\n', ' ').replace('\r', '')
doc = ' '.join(doc.split())
print(doc)


# In[304]:


tokenized_doc = nltk.word_tokenize(doc)
 
# tag sentences and use nltk's Named Entity Chunker
tagged_sentences = nltk.pos_tag(tokenized_doc)
ne_chunked_sents = nltk.ne_chunk(tagged_sentences)
 
# extract all named entities
named_entities = []
for tagged_tree in ne_chunked_sents:
    if hasattr(tagged_tree, 'label'):
        entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #
        entity_type = tagged_tree.label() # get NE category
        named_entities.append((entity_name, entity_type))
print(named_entities)


# In[275]:


### recognizing PERSONs from NLTK entities

per_nltk = []
for i in named_entities:
    if i[1]=='PERSON':
        if i[0] not in per_nltk:
            per_nltk.append(i[0])

print(per_nltk)


# In[288]:


### lowering them
per_nltk = [x.lower() for x in per_nltk]
for i in per_nltk:
    if i.startswith('emp') or i.startswith('agr'):
        per_nltk.remove(i)

print(per_nltk)


# In[274]:


### recognizing ORGANIZATIONS from NLTK entities
org_nltk = []
for i in named_entities:
    if i[1]=='ORGANIZATION':
        if i[0] not in org_nltk:
            org_nltk.append(i[0])

org_nltk = [x.lower() for x in org_nltk]
for i in org_nltk:
    if i.startswith('emp') or i.startswith('agr'):  ### we'll ignore words with emp or agr as they're not useful.
        org_nltk.remove(i)

print(org_nltk)


# ## NOW USING SPACY FOR NER

# In[245]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import re


# In[277]:


doc = open("document-analytics-master/document-analytics-master/employment contracts/"+file_name, "r", encoding="utf8")

doc = doc.read()


# In[278]:


doc = doc.replace('\n', ' ').replace('\r', '')
doc = ' '.join(doc.split())
print(doc)


# In[308]:


### NOW using SPACY for NER

from spacy import displacy
 
doc = nlp(doc)
displacy.render(doc, style='ent', jupyter=True)


# In[309]:


### checking the document type

amendment = ['Amendment to Employment Agreement','AMENDMENT', 'AMENDMENT TO EMPLOYMENT AGREEMENT']
doc_type = "Employment Agreement"
for sent in doc.sents:
    for i in amendment:
        if i in str(sent):
            doc_type = "AMENDMENT"
            break
print (doc_type)


# In[310]:


print([(X.text, X.label_) for X in doc.ents])


# In[311]:


### recognizing ORGANIZATIONS and PERSONS by SPACY

org= []
for ent in doc.ents:
    if ent.label_=='ORG':
        if ent.text not in org:
            org.append(ent.text)
print (org,'\n')

per= []
for ent in doc.ents:
    if ent.label_=='PERSON':
        if ent.text not in per:
            per.append(ent.text)
print (per)


# In[312]:


per = [x.lower() for x in per]
for i in per:
    if i.startswith('emp'):
        per.remove(i)
print (per)


# In[313]:


### finding the common PERSONS recognized by both NLTK and SPACY
lst3 = [value for value in per if value in per_nltk] 
print(lst3)


# In[290]:


### Name of the employee
employee = lst3[0]
employee = employee.upper()
print (employee)


# In[291]:


### Finding the employer name.

employee_first = employee.lower().split()[0]
#employee_second = employee.lower().split()[1]

#for i in lst3:
#    if i.startswith(employee_first) or i.startswith(employee_second):
#        lst3.remove(i)
#print (lst3)
#employer = lst3[-1]    ### We take -1 because the letter always end with the signature of the employer, so there is good chance of recognizing the employer name at the end.
#employer = employer.upper()
employer = per_nltk[-1].upper()
employer


# In[292]:


org = [x.lower() for x in org]
for i in org:
    if i.startswith('emp') or i.startswith('agr'):
        org.remove(i)

print(org)


# In[293]:


### Finding the common organizations recognized by both NLTK and Spacy.

org = [value for value in org if value in org_nltk]
print (org)


# In[294]:


### Finding the job profile of the employee being hired.
roles = ['Director of Operations', 'President and Chief Executive Officer', 'Chief  Executive  Officer','Senior Vice President','Vice President', 'Chief Financial Officer','Chief Human Resources Officer','Chairman of the Board of Directors','Chief Supply Chain Officer','Chief Creative Officer','Chief Procurement Officer','Contract CFO','Senior Engineer','Co-Chief Investment Officer']
role = ''
for sent in doc.sents:
    for i in roles:
        if i in str(sent):
            if i not in role:
                role = i+ ', '+ role
print (role)


# In[295]:


### Calculating the base salary per annum. We are taking the maximum here because yearly salary is always higher than monthly salary.
sal_list = []
for sent in doc.sents:
    sent = str(sent).split()
    for ind in sent:
        if "$" in ind:
            sal_list.append(ind)

base_sal = max(sal_list)
    
print (base_sal)


# In[296]:


### Creating the pandas DataFrame with the data so obtained.
data = ({'file':[doc_type],'Employer Name':[employer],'Employee Name':[employee], 'Role/Tile of the employee':[role], 'Base Salary(yearly)':[base_sal]})
ner = pd.DataFrame(data)


# In[297]:


print (ner)


# In[298]:


### making the csv

ner.to_csv(r'letter.csv')


# ### FEATURE EXTRACTION- MD SAMEEM ALI
