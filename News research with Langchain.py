#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install Packages
get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install sentence-transformers')
get_ipython().system('pip3 install unstructured libmagic python-magic python-magic-bin')


# In[1]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
pd.set_option('display.max_colwidth', 100)
from sentence_transformers import SentenceTransformer
import faiss
from langchain.document_loaders import SeleniumURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import pickle
from langchain.chains import RetrievalQAWithSourcesChain


# In[2]:


loader=SeleniumURLLoader(urls=["https://hindenburgresearch.com/adani/"])


# In[3]:


data=loader.load()


# In[4]:


data


# In[5]:


data_as_strings = [str(doc) for doc in data]
resulting_string = "\n".join(data_as_strings)
resulting_string


# In[6]:


splitter=RecursiveCharacterTextSplitter(
    separators=["\n"," "],
    chunk_size=1000,
    chunk_overlap=500)


# In[7]:


chunks=splitter.split_text(resulting_string)


# In[8]:


chunks[0]


# In[9]:


len(chunks)


# In[10]:


encoder=SentenceTransformer("all-mpnet-base-v2")
vectors=encoder.encode(chunks)


# In[11]:


vectors.shape


# In[12]:


dim=vectors.shape[1]
dim


# In[13]:


import faiss
index=faiss.IndexFlatL2(dim)


# In[14]:


index.add(vectors)
type(index)


# In[47]:


search_query="What was Samir vora reponsible for ?"
search_query_vector=encoder.encode(search_query)
search_query_vector.shape


# In[48]:


import numpy as np
new_search_query_vector=np.array(search_query_vector).reshape(1,-1)
new_search_query_vector.shape


# In[49]:


distances, I=index.search(new_search_query_vector,k=1)
I


# In[50]:


chunks[315]


# In[51]:


extracted_number = I[0, 0]

a=extracted_number
a


# In[52]:


searching_chunk=chunks[a]
searching_chunk


# In[53]:


file_path="index.pkl"
with open(file_path,"wb") as f:
    pickle.dump(index,f)


# In[54]:


if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorIndex=pickle.load(f)


# In[55]:


from langchain.llms import GooglePalm
api_key="AIzaSyAKH8XfZ0iUAo1yzwdQTTrbw-S1K6Q8nKQ"
llm=GooglePalm(google_api_key=api_key, temperature=0.4)


# In[56]:


question="What was Samir vora reponsible for ?"


# In[57]:


from langchain import PromptTemplate
template="""take below text as context and answer, the text is {context},the question is {query}"""
prompt=PromptTemplate(input_variables=['context','query'],template=template)
my_prompt=prompt.format(context=searching_chunk,query=question)


# In[58]:


my_prompt


# In[59]:


llm(my_prompt)


# In[ ]:




