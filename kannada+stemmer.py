
# coding: utf-8

# In[5]:

from nltk.tokenize import word_tokenize
import glob
import numpy as np
import pandas as pd


# In[72]:

suffix=['ನು','ರು','ನನ್ನು','ರನ್ನು','ನಿಂದ','ರಿಂದ','ನಿಗೆ','ರಿಗೆ','ನದೆಸೆಯಿಂದ','ರದೆಸೆಯಿಂದ','ರ','ನಲ್ಲಿ','ರಲ್ಲಿ',
  'ನಿಂದಿರು','ಇಂದರನ್ನು','ನಿಂದ','ದಿರಿಂದ','ದಿರಿಗೆ','ನದೆಸೆಯಿಂದ','ಇಂದಿರದೆಸೆಯಿಂದ','ನ',
   'ಇಂದಿರ','ನಲ್ಲಿ','ಇಂದಿರಲ್ಲಿ','ಳು','ರು','ಳನ್ನು','ರನ್ನು','ಳಿಂದ','ರಿಂದ','ಳಿಗೆ','ರಿಗೆ','ಳಾದೆಸೆಯಿಂದ',
   'ರದೆಸೆಯಿಂದ','ಳ','ರ','ಳಲ್ಲಿ','ರಲ್ಲಿ','ಳು','ನಿಂದಿರು','ಳನ್ನು','ಇಂದರನ್ನು','ಳಿಂದ','ದಿರಿಂದ','ಳಿಗೆ',
   'ನಿಂದಿರಿಗೆ','ಳಾದೆಸೆಯಿಂದ','ಇಂದಿರದೆಸೆಯಿಂದ','ಳ','ಇಂದಿರ','ಳಲ್ಲಿ','ಇಂದಿರಲ್ಲಿ','ವು','ಗಳು','ವನ್ನು',
   'ಗಳನ್ನು','ದಿಂದ','ಗ','ಳಿಂದ','ಕ್ಕೆ','ಗಳಿಗೆ','ದದೆಸೆಯಿಂದ','ಗಳದೆಸೆಯಿಂದ','ದ','ಗಳ','ದಲ್ಲಿ','ಗಳಲ್ಲಿ',
   'ದು','ವು','ದನ್ನು','ವುಗಳನ್ನು','ದಿಂದ','ವುಗಳಿಂದ','ದಿಕ್ಕೆ','ವುಗಳಿಗೆ','ದರದೆಸೆಯಿಂದ','ವುಗಳದೆಸೆಯಿಂದ',
   'ವ','ವುಗಳ','ದಲ್ಲಿ','ವುಗಳಲ್ಲಿ','ಯು','ಗಳು','ಯನ್ನು','ಗಳನ್ನು','ಯಿಂದ','ಗಳಿಂದ','ಗೆ','ಗಳಿಗೆ',
   'ಯದೆಸೆಯಿಂದ','ಗಳದೆಸೆಯಿಂದ','ಯ','ಗಳ','ಯಲ್ಲಿ','ಗಳಲ್ಲಿ','ಯು','ಯರು','ಯನ್ನು','ಯರನ್ನು',
   'ಯಿಂದ','ಯರಿಂದ','ಗೆ','ಯರಿಗೆ','ಯದೆಸೆಯಿಂದ','ಯರದೆಸೆಯಿಂದ','ಯ','ಯರ','ಯಲ್ಲಿ','ಯರಲ್ಲಿ',
   'ಯು','ಅಂದಿರು','ಯನ್ನು','ಅಂದಿರನ್ನು','ಯಿಂದ','ಗೆ','ಅಂದಿರಿಗೆ','ಯದೆಸೆಯಿಂದ','ಅಂದಿರದೆಸೆಯಿಂದ',
   'ಯ','ಯರ','ಯಲ್ಲಿ','ಅಂದಿರಲ್ಲಿ','ವು','ಗಳು','ವನ್ನು','ಗಳನ್ನು','ದಿಂದ','ಗಳಿಂದ','ಕ್ಕೆ','ಗಳಿಗೆ','ದದೆಸೆಯಿಂದ',
   'ಗಳದೆಸೆಯಿಂದ','ದ','ಗಳ','ದಲ್ಲಿ','ಗಳಲ್ಲಿ','ಉ','ಗಳು','ಅನ್ನು','ಗಳನ್ನು','ಯಿಂದ','ಗಳಿಂದ','ಗೆ','ಗಳಿಗೆ',
   'ನದೆಸೆಯಿಂದ','ಗಳದೆಸೆಯಿಂದ','ನ','ಗಳ','ಗಳು','ನಲ್ಲಿ','ಗಳಲ್ಲಿ','ಯು','ಯಂದಿರು','ಯನ್ನು','ಯಂದಿರನ್ನು','ಯಿಂದ',
   'ಯಂದಿರನ್ನು','ಗೆ','ಯಿಂದಿರಿಗೆ','ಯದೆಸೆಯಿಂದ','ಯಿಂದಿರದೆಸೆಯಿಂದ','ಯ','ಯಂದಿರ','ಯಲ್ಲಿ','ಯಂದಿರ','ವಾಗಿ',

]

prefix= ["ಪ್ರ","ಪ್ಯಾರಾ","ಅಪ","ಸ್ಯಾಮ್","ಆವಾ","ನಿಸ್","ನೀರ್","ದುಸ್","ಅಭಿ","ಪ್ರತಿ","ಪರಿ","ಉಪ",
        "ಆ","ವಿ","ಅಧಿ","ಅತಿ","ಉಥ್","ಸು","ದುರ್","ಅಣು","ಅತಿ","ನೀ","ಕು"]



# In[82]:

import math
import numpy as np
import glob
import collections
import string

docs=dict()


'''documents = ['ಸಾಧಾರಣವಾಗಿ ಹಾಡು, ಆಟಗಳನ್ನು ಕೂಡಿದ ಕೆಲವು ಘಂಟೆಗಳ ಬಳಿಕ ರಜೆ ಘೋಷಿಸಲಾಗುತ್ತದೆ ಹಾಗೂ',
             'ಸಾಂಸ್ಕೃತಿಕ ಮಕ್ಕಳ ಸಾಧಾರಣವಾಗಿ ದಿನಾಚರಣೆಯಂದು ಸಾಧಾರಣವಾಗಿ ವಿವಿಧೆಡೆ ಕಲಾ ಹಾಗೂ',
            'ಸಾಂಸ್ಕೃತಿಕ ರಜೆ ಕಾರ್ಯಕ್ರಮಗಳನ್ನು ಹಮ್ಮಿಕೊಳ್ಳಲಾಗುತ್ತದೆ.', 
             'ಚಿತ್ರಕಲೆ ಸ್ಪರ್ಧೆಗಳು ಪ್ರಮುಖ ರಜೆ',
             ]'''
query="ಹೂವಿನಲ್ಲಿ ಬಟ್ಟಲಿನಾಕಾರದ"
qterms=np.array([])
qterms=np.append(qterms,query.split(" "))

documents=np.array([])
for file in glob.glob('kfile\*.txt'):
    #print(file)
    with open(file, 'r',encoding='utf-8') as f:
        documents=np.append(documents,f.readlines())
print(documents)
stdoc=dict()
count=0
for doc in documents:
    word = word_tokenize(doc)
    #print(word)
    stem=np.array([])
    for w in word:
        #print(w)
        for s in suffix:
            for p in prefix:
                if w.endswith(s)&w.startswith(p):
                    w=w.replace(s,'').replace(p,'')
                elif w.endswith(s):
                    w=w.replace(s,'')
                elif w.startswith(p):
                    w=w.replace(p,'')
        stem=np.append(stem,w)
    stdoc[count]=stem
    count=count+1
#print(stdoc[0])
for i in range(len(stdoc)):
    print('document',i,':\n',stdoc[i],'\n') 

stqterms=np.array([])
stem=np.array([])
for w in qterms:
        #print(w)
    for s in suffix:
        for p in prefix:
            if w.endswith(s)&w.startswith(p):
                w=w.replace(s,'').replace(p,'')
            elif w.endswith(s):
                w=w.replace(s,'')
            elif w.startswith(p):
                w=w.replace(p,'')
    stqterms=np.append(stqterms,w)
print(stqterms)


# In[83]:

doccounter=len(documents)
#print(doccounter)
#print(documents)
data=np.array(doccounter)
for i in range(doccounter):
    words=np.array([])
    #for c in string.punctuation:
        #data=documents[i].replace(c,"")
    words=stdoc[i]
    #print(words)
    DF = collections.defaultdict(int)
    IDF = dict()
    TFIDF=dict()
    #print(data[1])
    #data1[i] = data1[i].translate(string.punctuation)
      
    for word in words:
        DF[word] += 1 
    #print(DF)
    for word in DF:
        IDF[word] = math.log(doccounter / float(DF[word]))  
    #print(IDF)
    for word in DF:
        TFIDF[word]=float(DF[word])*float(IDF[word])
    #print(TFIDF)
    docs[i]=TFIDF
#print(docs[0])


rank=dict()
sum=np.zeros(doccounter)



print("\n\nTFIDF weights")
print(len(qterms))
for i in range(doccounter):
    for word in stqterms:
        sum[i]=sum[i]+(docs.get(i).get(word, 0))
        rank[i]=sum[i]
print((rank))
print("Query:",query,"\n")
for i in (sorted(rank,key=rank.get,reverse=True)):
    print("document ",i+1,'\n',documents[i])
#print(docs[1])


# In[ ]:




# In[ ]:



