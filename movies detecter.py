#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ast


# In[2]:


creadits = pd.read_csv("tmdb_5000_credits.csv")
creadits.shape


# In[66]:


movies = pd.read_csv("tmdb_5000_movies.csv")
movies.shape


# In[67]:


df = movies.merge(creadits)
df


# In[68]:


df.info()


# In[69]:


#importent columns 
#genres,keywords,original_title,production_companies,cast,crew

movies_df = df[['id','genres','keywords','original_title','production_companies','overview','cast','crew']]
movies_df


# In[70]:


movies_df.isnull().sum()


# In[71]:


movies_df.dropna(inplace=True)


# In[7]:


movies_df['genres'][0]


# In[72]:


def tagname(tag):
    names = []
    for i in ast.literal_eval(tag):       
        names.append(i['name'])
    return names
# tagname([{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}])


# In[73]:


movies_df['genres'] = movies_df['genres'].apply(tagname)


# In[74]:


movies_df['keywords']=movies_df['keywords'].apply(tagname)
movies_df['keywords']


# In[75]:


movies_df


# In[76]:


movies_df['production_companies'] = movies_df['production_companies'].apply(tagname)


# In[77]:


movies_df.head()


# In[78]:


def castconvert(obj):
    L = []
    convert = 0
    for i in ast.literal_eval(obj):
        if convert!=3:
           L.append(i['name'])
           convert+=1
        else:
            break
    return L


# In[79]:


movies_df['cast']=movies_df['cast'].apply(castconvert)
movies_df['cast']


# In[80]:


movies_df.head()


# In[81]:


def featch_dir(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if(i['job']=='Director'):
            L.append(i['name'])
            break
    return L


# In[83]:


movies_df['crew']=movies_df['crew'].apply(featch_dir)


# In[84]:


movies_df


# In[62]:


movies_df['genres']=movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[85]:


movies_df['keywords']=movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
# movies_df['original_title']=movies_df['original_title'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['production_companies']=movies_df['production_companies'].apply(lambda x:[i.replace(" ","") for i in x])


# In[86]:


movies_df


# In[64]:


def overviewadd(obj):
    L =[]
    new = obj.split()
#     for i in obj:
#              L.split()
    return new
# overviewadd("In the 22nd century, a paraplegic Marine is")


# In[ ]:


movies_df['genres'] = movies_df['genres'].apply(lambda x: " ".join(x))


# In[87]:


movies_df['keywords'] = movies_df['keywords'].apply(lambda x: " ".join(x))
movies_df['production_companies'] = movies_df['production_companies'].apply(lambda x: " ".join(x))
movies_df['cast'] = movies_df['cast'].apply(lambda x: " ".join(x))
movies_df['crew'] = movies_df['crew'].apply(lambda x: " ".join(x))


# In[94]:


movies_df['genres'] = movies_df['genres'].apply(lambda x: " ".join(x))


# In[95]:


movies_df


# In[91]:


movies_df['genres']=movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[96]:


movies_df['tags'] = movies_df['genres']+movies_df['keywords']+movies_df['production_companies']+movies_df['overview']+movies_df['cast']+movies_df['crew']


# In[97]:


movies_df['tags']


# In[150]:


final = movies_df[['id','original_title','tags']]


# In[151]:


final.iloc[]


# In[100]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=5000,stop_words='english')


# In[105]:


cv.fit_transform(final['tags']).toarray().shape


# In[106]:


vectores=cv.fit_transform(final['tags']).toarray()
vectores


# In[110]:


import nltk


# In[113]:


from nltk.stem.porter import PorterStemmer


# In[114]:


ps = PorterStemmer()


# In[115]:


def stemper(text):
    L = []
    for i in text.split():
        L.append(ps.stem(i))
    return " ".join(L)


# In[116]:


final['tags'] = final['tags'].apply(stemper)


# In[117]:


final['tags']


# In[120]:


from sklearn.metrics.pairwise import cosine_similarity


# In[121]:


cosine_similarity(vectores)


# In[122]:


similerty = cosine_similarity(vectores)


# In[123]:


similerty[0]


# In[128]:


sorted(enumerate(similerty[0]),reverse=True,key=lambda x:x[1])[1:6]


# In[178]:


new=final[final.index==1216]
new.original_title


# In[204]:


def recommend(movie):
    L = []
    movies_index = final[final['original_title']==movie].index[0]
    
    distances = similerty[movies_index]
    #print(distances)
    movies_list = sorted(enumerate(distances),reverse=True,key=lambda x:x[1])[1:6]
    #print(movies_list)
    for i in movies_list:
        new=final[final.index==i[0]]
        L.append(new['original_title'])
    return L


# In[206]:


film=recommend('Spectre')
film


# In[ ]:




