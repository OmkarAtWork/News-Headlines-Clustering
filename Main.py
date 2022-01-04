import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from string import punctuation
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

pd.options.display.max_colwidth = 100

from google.colab import files
import io
import pandas as pd
import numpy as np

uploaded=files.upload()
df=pd.read_excel(io.BytesIO(uploaded['ScrapedData.xlsx']))
newsdf=df

pd.set_option('display.max_columns',None)
print(newsdf.head())

pd.options.display.max_colwidth = 100
newsdf = newsdf.drop(['Unnamed: 0'], axis=1)
newsdf.rename(columns={'0': 'headline'}, inplace=True)
print(newsdf)

#WHEN THE COLUMN NAME IS AN INTEGER
#newsdf.rename(columns={0: 'headline'}, inplace=True)
#print(newsdf)

#news_df=newsdf['headline']
news_df=newsdf
print(news_df.head())

news_cat=news_df
#news_cat['headline']=news_df
print(news_cat)

news_cat['headline'].head()


stop_words=set(stopwords.words('english'))
print(stop_words)

news_cat=news_cat.dropna()

news_cat['headline']=news_cat['headline'].str.lower()
print(news_cat['headline'])
news_cat.isnull().sum()

news_cat['headline'] =news_cat['headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
print(news_cat.head())
print(news_cat.shape)


#change text to lower case and removal of white spaces
import re
lower_train_text = []
for i in range(0,len(news_cat)):
  s = str(news_cat.iloc[i])
  s1 = s.strip()
  lower_train_text.append(s1.lower())
  #except KeyError:
    #print("Found at ",X_train[i])
   # print("")
punc_train_text = []
for i in range(0,len(lower_train_text)):
  s2 = (lower_train_text[i])
  s3 = re.sub(r'[^\w\s2]',"",s2)
  punc_train_text.append(s3)
  
#punc_train_text[4]
try11=punc_train_text

print("Before slicing:\n")
for i in range(0,5):
  print(punc_train_text[i])

for i in range(0,len(try11)):
  try11=punc_train_text[i]
  punc_train_text[i]=try11[12:len(try11)-20]

print("After slicing:\n")
for i in range(0,5):
  print(punc_train_text[i])
  
from sklearn.feature_extraction.text import TfidfVectorizer

train_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True,strip_accents='ascii', stop_words='english')
news_cat_tfidf = train_vectorizer.fit_transform(punc_train_text)

print(news_cat_tfidf.shape)
type(news_cat_tfidf)
dtv=news_cat_tfidf.toarray()
print(f"Number of Observations: {dtv.shape[0]}\nTokens/Features: {dtv.shape[1]}")

# Let's see an sample that has been preprocessed
dtv[1]


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def cluster_texts(num_clusters,tfidf):
  print('Beginning KMeans Clustering, number of clusters=',num_clusters,'\n')
  km=KMeans(n_clusters=num_clusters,max_iter=100,verbose=2,n_init=1).fit(tfidf)
  return km
'''
def elbow(tfidf):
  K=range(1,15)
  wss=[]
  sscore=[]
  for k in K:
    km=KMeans(n_clusters=k,n_init=10)
    kmeans=km.fit(tfidf)
    #score = silhouette_score(tfidf, km.labels_, metric='euclidean')
    #sscore.append(score)
    wss_itr=kmeans.inertia_
    wss.append(wss_itr)
  print("WSS:\n",wss)
  #print("Silhouetter:\n",score)
  #no_of_K=50
  plt.plot(K,wss,'bx-')
  plt.xlabel('no of k')
  plt.ylabel('wss')
  plt.show()

 elbow(dtv)
 '''

documents_vectorized=dtv
n=12
kmeans12=cluster_texts(n,documents_vectorized)

kmeans_df=pd.DataFrame()
kmeans_df['kmeans12']=kmeans12.labels_
kmeans_df['stemmed']=punc_train_text

print(news_cat.shape)
print(kmeans_df.shape)

kmeans_df['kmeans12'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='kmeans12',data=kmeans_df)
plt.show()

def common_words(df,df_column,num_words):
  cw=[]
  for i in range(0,12):
    common=Counter("".join(df.loc[df_column==i]['stemmed']).split()).most_common(num_words)
    for j in common:
      dict_={}
      dict_['cluster']=i
      dict_['word']=j[0]
      cw.append(dict_)
  return cw

from collections import Counter
print(kmeans_df.head(10))


#FINDING THE SIMILAR SENTENCES FROM A CLUSTER

for i in range(n):
  print("Cluster ",i,":")
  df_clustered=kmeans_df[kmeans_df['kmeans12']==i]
  print(df_clustered['stemmed'])
  print("\n\n")
  

#Common words
common_words(kmeans_df,kmeans_df['kmeans12'],10)
