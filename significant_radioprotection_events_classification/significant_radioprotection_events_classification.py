#Evenement significatif Analyse en radiothérapie externe

#Import Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk 
import tensorflow as tf

#1.Definir un objectif mesurable :
    #load data
    df = pd.read_excel('df.xlsx')
    #Objectif : Predire le grade de l'évenement significatif à partir de la description de l'évenement.
    #Objectif 2 : Ressortir de la description la localisation, la date et le titre.
    #Métrique : Score F1 -> 80%
    
#2.Exploration des données
    
#Analyse de forme

    #All row and columns showed
    pd.set_option('display.max_row', 82) 
    pd.set_option('display.max_column', 82)
    
    #Explore first fives rows ==> df values
    df.head()
    df.columns
    
    #target identification : Grade

    #shape : 82,6
    df.shape      
    
    #variable types : objet et int64 
    df.dtypes.value_counts()
    
    #date type 
    df['Date'].astype
    #remove 'publié le'
    df['Date'] = df.Date.str.lstrip('Publié le')
    #obj to datatime
    df['Date'] = pd.to_datetime(df['Date'])
    #date=index
    df = df.set_index('Date')
    df.head
    
    
    #Nan values : Absence de NAN value 
    sns.heatmap(df.isna(), cbar=False)
    
#Analyse de fond
    
    #drop columns
    df = df.drop('Unnamed: 0', axis=1)
    
    #target visualization_ 74,4% of grade 2 et 24,4% of grade 1 et 1,2% of grade 3 
    grade_distribution = df['Grade'].value_counts(normalize = True)
     
    #grade distribution circular diagram 
    labels = 'grade 2', 'grade 1', 'grade 3'
    sizes = grade_distribution
    explode = (0, 0, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    plt.show()   
    
    #Number events variation over time   
    # Add columns with year, month, and weekday name
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Weekday'] = df.index.weekday_name
    df.head
    
    
    #circular diagram
    df['Year'].value_counts(normalize = True).plot.bar()
    df['Month'].value_counts(normalize = True).plot.bar()
    df['Weekday'].value_counts(normalize = True).plot.bar()
    
    # Localisation distribution circular diagram 
    df['Localisation'].value_counts()[df['Localisation'].value_counts()>=3].plot.bar()
    
  
    #Remove grade 3
    indexN = df[df['Grade'] == 3 ].index
    df.drop(indexN , inplace=True)
    
    df['Grade'].value_counts(normalize = True).plot.bar()

    #Number of words in each messages
    df['word_count'] = df['Description'].str.split().str.len()
    
    # Visualize the distribution of word counts in each category
    sns.distplot(df[df['Grade']==1]['word_count'], label='grade1')
    sns.distplot(df[df['Grade']==2]['word_count'], label='grade2')
    plt.legend()
    plt.show()


#3.Preprocessing       
df = df.drop(['Description', 'Localisation', 'Year', 'Month','Weekday', 'word_count'], axis=1)

#Split to train and test set 
from sklearn.model_selection import train_test_split 
train_set, test_set = train_test_split(df, test_size=0.1, random_state=0)

train_set.shape
test_set.shape

#Text Cleaning
import nltk
from nltk.corpus import stopwords
import string

#process train
def process_text(text):
    
    #Remove Punctuationa
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #Remove Stop Words
    s = stopwords.words('french')
    clean_words = [word for word in nopunc.split() if word.lower() not in s]
    
    #Return a list of clean words
    return clean_words

#Tokenization and convert the text into a matrix of token counts
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer=process_text, stop_words='french', max_features=44)
vectorizer2 = TfidfVectorizer(analyzer=process_text, stop_words='french', max_features=44)

X_train_vect = vectorizer.fit_transform(train_set['Titre'])
X_test_vect = vectorizer2.fit_transform(test_set['Titre'])

X_train_vect.shape
X_train = vectorizer.get_feature_names()

X_test_vect.shape
X_test = vectorizer2.get_feature_names()


#4.Modelling      
from sklearn.cluster import KMeans

#determine numbre of cluster
inertia = []
K_range = range(1,44)
for k in K_range:
    model = KMeans(n_clusters=k).fit(X_train_vect)
    inertia.append(model.inertia_)

plt.plot(K_range, inertia)
plt.xlabel('nombre de clusters')
plt.ylabel('cout du modele (Inertia)')

#Model
model = KMeans(n_clusters=2)

#5.Procédure d'évaluation
#fit data and model
model.fit(X_train_vect)
 
#determined centroids
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

#determined word in cluster
for i in range(2):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :30]:
        print(' %s' % X_train[ind])

#predict model 
model.predict(X_train_vect)

