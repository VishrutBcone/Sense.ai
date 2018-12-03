
# coding: utf-8

# In[165]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
from gensim.summarization.summarizer import summarize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk import word_tokenize, sent_tokenize, ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob, Word
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pickle
import re
import os.path
import itertools
import datetime
import datefinder
import googlemaps
import numpy as np
gmaps = googlemaps.Client(key='AIzaSyCIvP02kj-vNt4KU4OZ0F5qGqt19ibv4jQ')


# # Scraping

# In[102]:


def get_lat_lng(location_name):
    geocode_result = gmaps.geocode(location_name)
    latitude = geocode_result[0]['geometry']['location']['lat']
    longitude = geocode_result[0]['geometry']['location']['lng']
    return latitude, longitude


# In[103]:


def get_only_text(urll,USER_AGENT={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}):
    page = requests.get(urll,headers = USER_AGENT)
    soup = BeautifulSoup(page.text, "lxml")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return(text)


# In[1]:


def scraping_weather(home_url,month_,day_,USER_AGENT={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}):
    url =  home_url + "/news?pg="
    link_ = []
    hline_ = []
    date_ = []
    Summary_ = []
    Content_ = []
    Sentiment_ = []
    wordnet_lemmatizer = WordNetLemmatizer()
    sid = SentimentIntensityAnalyzer()
    for page_num in range(1,11):
        url_data = requests.get(url+str(page_num),headers=USER_AGENT)
        html = BeautifulSoup(url_data.text,"lxml")
        header_data = html.findAll("div",{'class':'styles__listGroupItem__3RUmQ'})
        for hd_dt in header_data:
            link = hd_dt.find("a",href = True)
            hline = hd_dt.find("span",class_="styles__headline__1WDSw")
            date = hd_dt.find("span",class_="styles__wxTitleWrapTimestamp__12-cd")
            if date.text.split(',')[0] == (month_+" "+str(day_)):
                link_.append(home_url + link['href'])
                hline_.append(hline.text.strip())
                date_.append(date.text.strip())
                text_data = get_only_text(home_url + link['href'])
                words = sent_tokenize(text_data)
                freqTable = dict()
                words = [word.lower() for word in words]
                #words = [word for word in words if word.isalnum()]
                words = [wordnet_lemmatizer.lemmatize(word) for word in words]
                tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3),
                                 stop_words = 'english')
                tfidf_matrix =  tf.fit_transform(words)
                idf = tf.idf_
                dict_idf = dict(zip(tf.get_feature_names(), idf))
                sentences = sent_tokenize(text_data)
                sent_list = set()
                for sentence in sentences:
                    for value, term in dict_idf.items():
                        if value in sentence:
                            sent_list.add(sentence)
                Content_.append(text_data)
                scores = sid.polarity_scores("\n".join(sent_list))
                Summary_.append(summarize("\n".join(sent_list),ratio = 0.2))
                Sentiment_.append(scores['compound'])
    data = pd.DataFrame({"Date":date_,"url":link_, "Headline":hline_,"Content":Content_,"Summary":Summary_,"Sentiments":Sentiment_})
    return(data)


# In[2]:


def scraping_acc_weather(home_url,month_,day_,USER_AGENT={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}):
    url =  home_url + "en/weather-news"
    link_ = []
    hline_ = []
    date_ = []
    Summary_ = []
    Content_ = []
    Sentiment_ = []
    for page_num in range(1,11):
        url_data = requests.get(url + "?page=" + str(page_num),headers=USER_AGENT)
        data = BeautifulSoup(url_data.text,"lxml")
        wordnet_lemmatizer = WordNetLemmatizer()
        sid = SentimentIntensityAnalyzer()
        headlines = data.findAll("div",{'class':'info'})
        for head in headlines:
            date = head.find("h5")
            if date != None:
                linkk = head.find("a")
                dte = date.text.strip().split(',')[0]
                try:
                    day__ = dte.split(' ')[1]
                    if (date.text.strip().split(',')[0][:3] + " "+ str(day__)) == (month_+" "+str(day_)):
                        link_.append(linkk['href'])
                        date_.append(date.text.strip())
                        hline_.append(linkk.text)
                        text_data = get_only_text(linkk['href'])
                        words = sent_tokenize(text_data)
                        freqTable = dict()
                        words = [word.lower() for word in words]
                        #words = [word for word in words if word.isalnum()]
                        words = [wordnet_lemmatizer.lemmatize(word) for word in words]
                        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3),
                                         stop_words = 'english')
                        tfidf_matrix =  tf.fit_transform(words)
                        idf = tf.idf_
                        dict_idf = dict(zip(tf.get_feature_names(), idf))
                        sentences = sent_tokenize(text_data)
                        sent_list = set()
                        for sentence in sentences:
                            for value, term in dict_idf.items():
                                if value in sentence:
                                    sent_list.add(sentence)
                        Content_.append(text_data)
                        scores = sid.polarity_scores("\n".join(sent_list))
                        Summary_.append(summarize("\n".join(sent_list),ratio = 0.2))
                        Sentiment_.append(scores['compound'])
                except IndexError:
                    pass
    data = pd.DataFrame({"Date":date_,"url":link_, "Headline":hline_,"Content":Content_,"Summary":Summary_,"Sentiments":Sentiment_})
    return(data)


# In[3]:


def scraping_geopolitical(home_url,month_,day_,USER_AGENT={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}):
    url =  home_url + "category/asia-geopolitics"
    link_ = []
    hline_ = []
    date_ = []
    Summary_ = []
    Content_ = []
    Sentiment_ = []
    for page_num in range(1,11):
        url_data = requests.get(url + "/page/"+ str(page_num),headers=USER_AGENT)
        data = BeautifulSoup(url_data.text,"lxml")
        wordnet_lemmatizer = WordNetLemmatizer()
        sid = SentimentIntensityAnalyzer()
        headlines = data.findAll("div",{'class':'row'})
        for head in headlines:
            hd = head.find("div",class_="postPrevTitle")
            lnkk = head.find("a")
            date = head.find("div",class_="postPrevSubtitle")
            if date != None:
                dte = date.text.strip().split(',')[0]
                day__ = dte.split(' ')[1]
                if (date.text.strip().split(',')[0][:3] + " "+ str(day__)) == (month_+" "+str(day_)):
                    date_.append(date.text.strip())
                    hline_.append(hd.text.strip())
                    link_.append(lnkk['href'])
                    text_data = get_only_text(lnkk['href'])
                    words = sent_tokenize(text_data)
                    freqTable = dict()
                    words = [word.lower() for word in words]
                    #words = [word for word in words if word.isalnum()]
                    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
                    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3),
                                     stop_words = 'english')
                    tfidf_matrix =  tf.fit_transform(words)
                    idf = tf.idf_
                    dict_idf = dict(zip(tf.get_feature_names(), idf))
                    sentences = sent_tokenize(text_data)
                    sent_list = set()
                    for sentence in sentences:
                        for value, term in dict_idf.items():
                            if value in sentence:
                                sent_list.add(sentence)
                    Content_.append(text_data)
                    scores = sid.polarity_scores("\n".join(sent_list))
                    Summary_.append(summarize("\n".join(sent_list),ratio = 0.2))
                    Sentiment_.append(scores['compound'])
    data = pd.DataFrame({"Date":date_,"url":link_, "Headline":hline_,"Content":Content_,"Summary":Summary_,"Sentiments":Sentiment_})
    return(data)


# In[4]:


def geo_south_america(home_url,month_,day_,USER_AGENT={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}):
    url =  home_url + "Geopolitics/"
    link_ = []
    hline_ = []
    date_ = []
    Summary_ = []
    Content_ = []
    Sentiment_ = []
    for page_num in range(1,11):
        url_data = requests.get(url + "/Page-"+ str(page_num)+".html",headers=USER_AGENT)
        data = BeautifulSoup(url_data.text,"lxml")
        wordnet_lemmatizer = WordNetLemmatizer()
        sid = SentimentIntensityAnalyzer()
        headlines = data.findAll("div",{'class':'categoryArticle__content'})
        for head in headlines:
            hd = head.find("h2",class_="categoryArticle__title")
            lnkk = head.find("a")
            date = head.find("p",class_="categoryArticle__meta")
            if date != None:
                dte = date.text.strip().split(',')[0]
                day__ = dte.split(' ')[1]
                if (date.text.strip().split(',')[0] == month_+" "+str(day_)):
                    date_.append(date.text.strip())
                    hline_.append(hd.text.strip())
                    link_.append(lnkk['href'])
                    
                    text_data = get_only_text(lnkk['href'])
                    words = sent_tokenize(text_data)
                    freqTable = dict()
                    words = [word.lower() for word in words]
                    #words = [word for word in words if word.isalnum()]
                    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
                    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3),
                                     stop_words = 'english')
                    tfidf_matrix =  tf.fit_transform(words)
                    idf = tf.idf_
                    dict_idf = dict(zip(tf.get_feature_names(), idf))
                    sentences = sent_tokenize(text_data)
                    sent_list = set()
                    for sentence in sentences:
                        for value, term in dict_idf.items():
                            if value in sentence:
                                sent_list.add(sentence)
                    Content_.append(text_data)
                    scores = sid.polarity_scores("\n".join(sent_list))
                    Summary_.append(summarize("\n".join(sent_list),ratio = 0.2))
                    Sentiment_.append(scores['compound'])
    data = pd.DataFrame({"Date":date_,"url":link_, "Headline":hline_,"Content":Content_,"Summary":Summary_,"Sentiments":Sentiment_})
    return(data)


# In[5]:


def scraping_reuters(home_url,month_,day_,USER_AGENT={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}):
    url =  home_url + "?view=page&page=1&pageSize=10"
    link_ = []
    hline_ = []
    date_ = []
    Summary_ = []
    Content_ = []
    Sentiment_ = []
    for page_num in range(1,11):
        url_data = requests.get("https://www.reuters.com/news/archive/tsunami?view=page&page=" +str(page_num)+ "&pageSize=10",headers=USER_AGENT)
        data = BeautifulSoup(url_data.text,"lxml")
        wordnet_lemmatizer = WordNetLemmatizer()
        sid = SentimentIntensityAnalyzer()
        headlines = data.findAll("div",{'class':'story-content'})
        for head in headlines:
            hd = head.find("h3")
            lnkk = head.find("a")
            date = head.find("span")
            if date != None:
                dte = date.text.strip().split(',')[0]
                day__ = dte.split(' ')[1]
                if (date.text.strip().split(',')[0][:3] + " "+ str(day__)) == (month_+" "+str(day_)):
                    date_.append(date.text.strip())
                    hline_.append(hd.text.strip())
                    link_.append("https://www.reuters.com" + lnkk['href'])
                    text_data = get_only_text("https://www.reuters.com" + lnkk['href'])
                    words = sent_tokenize(text_data)
                    freqTable = dict()
                    words = [word.lower() for word in words]
                    #words = [word for word in words if word.isalnum()]
                    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
                    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3),
                                     stop_words = 'english')
                    tfidf_matrix =  tf.fit_transform(words)
                    idf = tf.idf_
                    dict_idf = dict(zip(tf.get_feature_names(), idf))
                    sentences = sent_tokenize(text_data)
                    sent_list = set()
                    for sentence in sentences:
                        for value, term in dict_idf.items():
                            if value in sentence:
                                sent_list.add(sentence)
                    Content_.append(text_data)
                    scores = sid.polarity_scores("\n".join(sent_list))
                    Summary_.append(summarize("\n".join(sent_list),ratio = 0.2))
                    Sentiment_.append(scores['compound'])
    data = pd.DataFrame({"Date":date_,"url":link_, "Headline":hline_,"Content":Content_,"Summary":Summary_,"Sentiments":Sentiment_})
    return(data)


# In[6]:


def eco_atimes(home_url,month_,day_,USER_AGENT={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}):
    url =  home_url + "tag/business/"
    link_ = []
    hline_ = []
    date_ = []
    Summary_ = []
    Content_ = []
    Sentiment_ = []
    for page_num in range(1,11):
        url_data = requests.get(url + "/page/" + str(page_num)+ "/",headers=USER_AGENT)
        data = BeautifulSoup(url_data.text,"lxml")
        wordnet_lemmatizer = WordNetLemmatizer()
        sid = SentimentIntensityAnalyzer()
        headlines = data.findAll("div",{'class':'item-content has-image'})
        for head in headlines:
            hd = head.find("div",class_="headline")
            lnkk = head.find("a")
            date = head.find("div",class_="date")
            if date != None:
                dte = date.text.strip().split(',')[0]
                day__ = dte.split(' ')[1]
                if (date.text.strip().split(',')[0]) == (month_+" "+str(day_)):
                    date_.append(date.text.strip())
                    hline_.append(hd.text.strip())
                    link_.append(lnkk['href'])
                    text_data = get_only_text(lnkk['href'])
                    words = sent_tokenize(text_data)
                    freqTable = dict()
                    words = [word.lower() for word in words]
                    #words = [word for word in words if word.isalnum()]
                    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
                    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3),
                                     stop_words = 'english')
                    tfidf_matrix =  tf.fit_transform(words)
                    idf = tf.idf_
                    dict_idf = dict(zip(tf.get_feature_names(), idf))
                    sentences = sent_tokenize(text_data)
                    sent_list = set()
                    for sentence in sentences:
                        for value, term in dict_idf.items():
                            if value in sentence:
                                sent_list.add(sentence)
                    Content_.append(text_data)
                    scores = sid.polarity_scores("\n".join(sent_list))
                    Summary_.append(summarize("\n".join(sent_list),ratio = 0.2))
                    Sentiment_.append(scores['compound'])
    data = pd.DataFrame({"Date":date_,"url":link_, "Headline":hline_,"Content":Content_,"Summary":Summary_,"Sentiments":Sentiment_})
    return(data)


# In[7]:


def dis_health(home_url,month_,day_,USER_AGENT={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}):
    url =  home_url + "category/news/"
    link_ = []
    hline_ = []
    date_ = []
    Summary_ = []
    Content_ = []
    Sentiment_ = []
    for page_num in range(1,11):
        url_data = requests.get(url + "page/" + str(page_num) + "/",headers=USER_AGENT)
        data = BeautifulSoup(url_data.text,"lxml")
        wordnet_lemmatizer = WordNetLemmatizer()
        sid = SentimentIntensityAnalyzer()
        headlines = data.findAll("div",{'class':'item-details'})
        for head in headlines:
            hd = head.find("h3",class_="entry-title td-module-title")
            lnkk = head.find("a")
            date = head.find("div",class_="td-module-meta-info")
            if date != None:
                month = date.text.strip().split(' ')[1][:3]
                day__ = date.text.strip().split(' ')[0][:2]
                if (month +" " +day__) == (month_+" "+str(day_)):
                    date_.append(date.text.strip())
                    hline_.append(hd.text.strip())
                    link_.append(lnkk['href'])
                    text_data = get_only_text(lnkk['href'])
                    words = sent_tokenize(text_data)
                    freqTable = dict()
                    words = [word.lower() for word in words]
                    #words = [word for word in words if word.isalnum()]
                    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
                    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3),
                                     stop_words = 'english')
                    tfidf_matrix =  tf.fit_transform(words)
                    idf = tf.idf_
                    dict_idf = dict(zip(tf.get_feature_names(), idf))
                    sentences = sent_tokenize(text_data)
                    sent_list = set()
                    for sentence in sentences:
                        for value, term in dict_idf.items():
                            if value in sentence:
                                sent_list.add(sentence)
                    Content_.append(text_data)
                    scores = sid.polarity_scores("\n".join(sent_list))
                    Summary_.append(summarize("\n".join(sent_list),ratio = 0.2))
                    Sentiment_.append(scores['compound'])
    data = pd.DataFrame({"Date":date_,"url":link_, "Headline":hline_,"Content":Content_,"Summary":Summary_,"Sentiments":Sentiment_})
    return(data)


# In[8]:


def dis_reuters(home_url,month_,day_,USER_AGENT={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}):
    url =  home_url + "?view=page&page=1&pageSize=10"
    link_ = []
    hline_ = []
    date_ = []
    Summary_ = []
    Content_ = []
    Sentiment_ = []
    for page_num in range(1,11):
        url_data = requests.get("https://in.reuters.com/news/archive/health?view=page&page=" +str(page_num)+ "&pageSize=10",headers=USER_AGENT)
        data = BeautifulSoup(url_data.text,"lxml")
        wordnet_lemmatizer = WordNetLemmatizer()
        sid = SentimentIntensityAnalyzer()
        headlines = data.findAll("div",{'class':'story-content'})
        for head in headlines:
            hd = head.find("h3",class_ = "story-title")
            lnkk = head.find("a")
            date = head.find("span")
            if date != None:
                month = date.text.strip().split(' ')[1]
                month = month.capitalize()
                day__ = date.text.strip().split(' ')[0]
                if (month +" " +day__ == (month_+" "+str(day_))):
                    date_.append(date.text.strip())
                    hline_.append(hd.text.strip())
                    link_.append("https://in.reuters.com" + lnkk['href'])
                    text_data = get_only_text("https://in.reuters.com" + lnkk['href'])
                    words = sent_tokenize(text_data)
                    freqTable = dict()
                    words = [word.lower() for word in words]
                    #words = [word for word in words if word.isalnum()]
                    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
                    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3),
                                     stop_words = 'english')
                    tfidf_matrix =  tf.fit_transform(words)
                    idf = tf.idf_
                    dict_idf = dict(zip(tf.get_feature_names(), idf))
                    sentences = sent_tokenize(text_data)
                    sent_list = set()
                    for sentence in sentences:
                        for value, term in dict_idf.items():
                            if value in sentence:
                                sent_list.add(sentence)
                    Content_.append(text_data)
                    scores = sid.polarity_scores("\n".join(sent_list))
                    Summary_.append(summarize("\n".join(sent_list),ratio = 0.2))
                    Sentiment_.append(scores['compound'])
    data = pd.DataFrame({"Date":date_,"url":link_, "Headline":hline_,"Content":Content_,"Summary":Summary_,"Sentiments":Sentiment_})
    return(data)


# In[112]:


def fetch_results(search_term, number_results, language_code = "en"):
    escaped_search_term = search_term.replace(' ', '+')

    google_url = 'https://www.google.com/search?q={}&num={}&hl={}'.format(escaped_search_term, number_results, language_code)
    response = requests.get(google_url, headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'})
    response.raise_for_status()

    return response.text


# # Modeling
# 

# In[113]:


def remove_stopWords(s):
        '''For removing stop words
        '''
        stop_words = set(stopwords.words('english'))
        s = ' '.join(word for word in s.split() if word not in stop_words)
        return s


# In[114]:


def preprocess(data):
    data.dropna(axis=0,inplace=True)
    #remove parethesses
    data['Feature'] = data.Feature.map(lambda x:re.sub(r'\([^)]*\)', '', x))
    
    #Lowercase
    data['Feature'] = data.Feature.apply(lambda x : str.lower(x))

    #Stopwords remove
    data["Feature"] = data.Feature.apply(lambda x: remove_stopWords(x))

    #Remove Punctuation(Speacial Characters)
    data['Feature'] = data['Feature'].map(lambda x: re.sub(r'\W+', ' ', x))
    
    #Lemmetization
    data['Feature'] = data['Feature'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return (data.sample(frac=1))


# In[115]:


def train_model(level,classifier, feature_vector_train, label, feature_vector_valid,tfidf_transformer,count_vect,valid_y,encoder,list_tst):
    # fit the training dataset on the classifier
    fit_cl = classifier.fit(feature_vector_train, label)
    with open(level + "_model.pckl", "wb") as f:
        pickle.dump(fit_cl, f)
        f.close()
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    validation = classifier.predict(tfidf_transformer.fit_transform(count_vect.transform(list_tst)))
    proba = classifier.predict_proba(tfidf_transformer.fit_transform(count_vect.transform(list_tst)))
    #print(classification_report(valid_y,predictions))
    return metrics.accuracy_score(predictions, valid_y),encoder.inverse_transform(validation), proba


# In[116]:


def modeling(level,data,Feature_col, Class_col,list_tst):
    valid_data_ = []
    count_vect = CountVectorizer(analyzer='word',max_df=0.85)
    word_vector = count_vect.fit_transform(data[Feature_col])
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    new_input = tfidf_transformer.fit_transform(count_vect.transform(data[Feature_col]))
    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    class_label = encoder.fit_transform(data[Class_col])
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(new_input,class_label,test_size=0.25, random_state=7010)
    
    accuracy,predict,proba = train_model(level,linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'), train_x,train_y,valid_x,tfidf_transformer,count_vect,valid_y,encoder,list_tst)
    print("Accuracy of Model with Count Vectorizer is : - ",accuracy)
    for i,j in zip(list_tst,predict):
        valid_data_.append([i,j])
    return valid_data_,accuracy, proba 


# In[117]:


def splitDataFrameList(df,target_column,separator):
    row_accumulator = []

    def splitListToRows(row, separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    df.apply(splitListToRows, axis=1, args = (separator, ))
    new_df = pd.DataFrame(row_accumulator)
    return new_df


# In[118]:


def scrape_(data):
    DataFrames = []
    testing_data = None
    used = []
    for inp in range(len(data)):
        date_ful = data["Date of Article"][inp]
        url_ = data["Source"][inp]
        month_ = date_ful.month_name()[:3]
        day_ = date_ful.day
        if len(str(day_)) == 1:
            day_ = "0"+str(day_)
        if (url_,date_ful) not in used:
            if url_ == "https://oilprice.com/":
                testing_data = geo_south_america(url_,month_,day_)
                testing_data.insert(loc=0,column="Source", value=[url_] * len(testing_data.Date))
            if url_ == "https://weather.com":
                testing_data = scraping_weather(url_,month_,day_)
                testing_data.insert(loc=0,column="Source", value=[url_] * len(testing_data.Date))
            elif url_ == "https://www.accuweather.com/":
                testing_data = scraping_acc_weather(url_,month_,day_)
                testing_data.insert(loc=0,column="Source", value=[url_] * len(testing_data.Date))
            elif url_ == "https://www.reuters.com/news/archive/tsunami":
                testing_data = scraping_reuters(url_,month_,day_)
                testing_data.insert(loc=0,column="Source", value=[url_] * len(testing_data.Date))
            elif url_ == "https://thediplomat.com/":
                testing_data = scraping_geopolitical(url_,month_,day_)
                testing_data.insert(loc=0,column="Source", value=[url_] * len(testing_data.Date))
            elif url_ == "http://www.atimes.com/":
                testing_data = eco_atimes(url_,month_,day_)
                testing_data.insert(loc=0,column="Source", value=[url_] * len(testing_data.Date))
            elif url_ == "https://www.healtheuropa.eu/":
                testing_data = dis_health(url_,month_,day_)
                testing_data.insert(loc=0,column="Source", value=[url_] * len(testing_data.Date))
            elif url_ == "https://in.reuters.com/news/archive/health":
                testing_data = dis_reuters(url_,month_,day_)
                testing_data.insert(loc=0,column="Source", value=[url_] * len(testing_data.Date))
            used.append((url_,date_ful))
            DataFrames.append(testing_data)
    return DataFrames


# In[172]:


ip = "15_days"


# In[174]:


Input_file = pd.read_excel("app_input_file.xlsx")
date_list_ = []
if ip == "today":
    for j in range(len(Input_file)):
        date_list= []
        current_time = datetime.date.today() - datetime.timedelta(days=1)
        date_list_.append(current_time.strftime('%m/%d/%Y'))
    Input_file["Date of Article"] = date_list_
    Input_file["Date of Article"] = pd.to_datetime(Input_file['Date of Article'])
    Input_file.drop_duplicates(inplace=True,keep="first")
    DataFrames = scrape_(Input_file)
    testing_data = pd.concat(DataFrames, axis = 0)
    testing_data.drop_duplicates(inplace=True,keep="first")
elif ip == "7_days":
    for j in range(len(Input_file)):
        date_list= []
        for i in range(1,8):
            current_time = datetime.date.today() - datetime.timedelta(days=i)
            date_list.append(current_time.strftime('%m/%d/%Y'))
        date_list_.append(date_list)
    Input_file["Date of Article"] = date_list_
    Input_file['Date of Article'] = Input_file['Date of Article'].astype(str).str[1:-1]
    Input_file_ = splitDataFrameList(Input_file,"Date of Article",",")
    Input_file_["Date of Article"] = pd.to_datetime(Input_file_['Date of Article'])
    Input_file_.drop_duplicates(inplace=True,keep="first")
    DataFrames = scrape_(Input_file_)
    testing_data = pd.concat(DataFrames, axis = 0)
    testing_data.drop_duplicates(inplace=True,keep="first")
elif ip == "15_days":
    for j in range(len(Input_file)):
        date_list= []
        for i in range(1,16):
            current_time = datetime.date.today() - datetime.timedelta(days=i)
            date_list.append(current_time.strftime('%m/%d/%Y'))
        date_list_.append(date_list)
    Input_file["Date of Article"] = date_list_
    Input_file['Date of Article'] = Input_file['Date of Article'].astype(str).str[1:-1]
    Input_file_ = splitDataFrameList(Input_file,"Date of Article",",")
    Input_file_["Date of Article"] = pd.to_datetime(Input_file_['Date of Article'])
    Input_file_.drop_duplicates(inplace=True,keep="first")
    DataFrames = scrape_(Input_file_)
    testing_data = pd.concat(DataFrames, axis = 0)
    testing_data.drop_duplicates(inplace=True,keep="first")
elif ip == "30_days":
    for j in range(len(Input_file)):
        date_list= []
        for i in range(1,31):
            current_time = datetime.date.today() - datetime.timedelta(days=i)
            date_list.append(current_time.strftime('%m/%d/%Y'))
        date_list_.append(date_list)
    Input_file["Date of Article"] = date_list_
    Input_file['Date of Article'] = Input_file['Date of Article'].astype(str).str[1:-1]
    Input_file_ = splitDataFrameList(Input_file,"Date of Article",",")
    Input_file_["Date of Article"] = pd.to_datetime(Input_file_['Date of Article'])
    Input_file_.drop_duplicates(inplace=True,keep="first")
    DataFrames = scrape_(Input_file_)
    testing_data = pd.concat(DataFrames, axis = 0)
    testing_data.drop_duplicates(inplace=True,keep="first")


# In[175]:


if os.path.isfile("scrapped_news.xlsx"):
    old_testing_data = pd.read_excel("scrapped_news.xlsx")
    testing_data_ = pd.concat([testing_data,old_testing_data]).reset_index(drop=True)
    testing_data_.drop_duplicates(inplace=True)
    testing_data_.to_excel("scrapped_news.xlsx",index=False)
else:
    testing_data.drop_duplicates(inplace=True)
    testing_data.to_excel("scrapped_news.xlsx",index=False)


# In[176]:


testing_data['Feature'] = testing_data[["Headline","Summary"]].apply(lambda x: "".join(x),axis = 1)
testing_data['Class_1'] = ""
testing_data['Class_2'] = ""
testing_data['Location'] = ""
testing_data['Dates'] = ""
testing_data = preprocess(testing_data)
list_test = list(testing_data["Feature"])


# In[177]:


# Read the input data and cLEAN IT UP FOR THE 1ST LEVEL
Input_data = pd.read_excel("Classification_data_Default.xlsx")
First_level = Input_data[["Headline","Summary","Class_1"]]
First_level['Feature'] = First_level[["Headline","Summary"]].apply(lambda x: "".join(x),axis = 1)
First_level = First_level.iloc[:,[3,2]].reset_index(drop=True)
First_level = preprocess(First_level)

# Read the input data and cLEAN IT UP FOR THE 2nd Level weather
Second_level_weather = Input_data[["Headline","Summary","Class_2","Class_1"]]
Second_level_weather['Feature'] = Second_level_weather[["Headline","Summary"]].apply(lambda x: "".join(x),axis = 1)
#Second_level_weather = Second_level_weather.iloc[:,[3,2]].reset_index(drop=True)
Second_level_weather = Second_level_weather.loc[Second_level_weather["Class_1"] == "Weather"][["Feature","Class_1","Class_2"]]
Second_level_weather = preprocess(Second_level_weather)

# Read the input data and cLEAN IT UP FOR THE 2nd Level disruptive
Second_level_disruptive = Input_data[["Headline","Summary","Class_2","Class_1"]]
Second_level_disruptive['Feature'] = Second_level_disruptive[["Headline","Summary"]].apply(lambda x: "".join(x),axis = 1)
Second_level_disruptive = Second_level_disruptive.loc[Second_level_disruptive["Class_1"] == "Disruptive"][["Feature","Class_1","Class_2"]]
Second_level_disruptive = preprocess(Second_level_disruptive)

# Read the input data and cLEAN IT UP FOR THE 2nd Level economic
Second_level_economic = Input_data[["Headline","Summary","Class_2","Class_1"]]
Second_level_economic['Feature'] = Second_level_economic[["Headline","Summary"]].apply(lambda x: "".join(x),axis = 1)
Second_level_economic = Second_level_economic.loc[Second_level_economic["Class_1"] == "Economic"][["Feature","Class_1","Class_2"]]
Second_level_economic = preprocess(Second_level_economic)

# Read the input data and cLEAN IT UP FOR THE 2nd Level geopolitics
Second_level_geopolitical = Input_data[["Headline","Summary","Class_2","Class_1"]]
Second_level_geopolitical['Feature'] = Second_level_geopolitical[["Headline","Summary"]].apply(lambda x: "".join(x),axis = 1)
Second_level_geopolitical = Second_level_geopolitical.loc[Second_level_geopolitical["Class_1"] == "Geopolitical"][["Feature","Class_1","Class_2"]]
Second_level_geopolitical = preprocess(Second_level_geopolitical)


# In[178]:


def Load_model(data,feature_col,model_name,list_tst,c_label):
    count_vect = CountVectorizer(analyzer='word',max_df=0.85)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    count_vect.fit_transform(data[feature_col])
    tfidf_transformer.fit_transform(count_vect.transform(data[feature_col]))
    loaded_model = pickle.load(open(model_name, 'rb'))
    validation = loaded_model.predict(tfidf_transformer.fit_transform(count_vect.transform(list_tst)))
    prob = loaded_model.predict_proba(tfidf_transformer.fit_transform(count_vect.transform(list_tst)))
    encoder = preprocessing.LabelEncoder()
    encoder.fit_transform(data[c_label])
    predictions = [ab for ab in zip(list_tst,encoder.inverse_transform(validation))]
    return predictions, prob


# In[179]:


predictions, prob_first = Load_model(First_level, "Feature", "First_level_training_model.pckl",list_test, "Class_1")
Classes = []
Wther = []
Disrpt = []
Geop = []
Econmn = []
Prob_ = []
for index, prd in enumerate(predictions):
    Prob_.append(max(prob_first[index]))
    if prd[1] == "Weather":
        Wther.append(prd[0])
    elif prd[1] == "Economic":
        Econmn.append(prd[0])
    elif prd[1] == "Geopolitical":
        Geop.append(prd[0])
    else:
        Disrpt.append(prd[0])


# In[180]:


predict_prob_we_ = []
predict_prob_dis_ = []
predict_prob_geo_ = []
predict_prob_eco_ = []
if len(Wther) > 0:
    We_pred, prob_2nd_level_wea = Load_model(Second_level_weather, "Feature", "Second_level_training_weather_model.pckl",Wther, "Class_2")
    for we in We_pred:
        testing_data.loc[testing_data['Feature'] == we[0], "Class_2"] = we[1]
    for i in prob_2nd_level_wea:
        predict_prob_we_.append(max(i))
if len(Geop) > 0:
    Geo_pred, prob_2nd_level_geo = Load_model(Second_level_geopolitical, "Feature", "Second_level_training_geopolitical_model.pckl",Geop, "Class_2")
    for ge in Geo_pred:
        testing_data.loc[testing_data['Feature'] == ge[0], "Class_2"] = ge[1]
    for j in prob_2nd_level_geo:
        predict_prob_geo_.append(max(j))
    
if len(Econmn) > 0:
    Eco_pred, prob_2nd_level_eco = Load_model(Second_level_economic, "Feature", "Second_level_training_economic_model.pckl",Econmn, "Class_2")
    for ec in Eco_pred:
        testing_data.loc[testing_data['Feature'] == ec[0], "Class_2"] = ec[1]
    for k in prob_2nd_level_eco:
        predict_prob_eco_.append(max(k))
if len(Disrpt) > 0:
    Dis_pred, prob_2nd_level_dis =Load_model(Second_level_disruptive, "Feature", "Second_level_training_disruptive_model.pckl",Disrpt, "Class_2")
    for dc in Dis_pred:
        testing_data.loc[testing_data['Feature'] == dc[0], "Class_2"] = dc[1]
    for l in prob_2nd_level_dis:
        predict_prob_dis_.append(max(l))


# In[181]:


for predc in predictions:
    testing_data.loc[testing_data['Feature'] == predc[0], "Class_1"] = predc[1]


# In[182]:


testing_data_new = testing_data[["Source","Headline","url","Summary","Class_1","Class_2"]]
testing_data_new["prob_class_1"] = Prob_
testing_data_new["prob_class_2"] = ""


# In[183]:


for ijkl in range(testing_data_new.shape[0]):
    if len(predict_prob_we_) > 0:
        testing_data_new.loc[testing_data_new['Class_1'] == "Weather", "prob_class_2"] = predict_prob_we_
    if len(predict_prob_dis_) > 0:
        testing_data_new.loc[testing_data_new['Class_1'] == "Disruptive", "prob_class_2"] = predict_prob_dis_
    if len(predict_prob_geo_) > 0:
        testing_data_new.loc[testing_data_new['Class_1'] == "Geopolitical", "prob_class_2"] = predict_prob_geo_
    if len(predict_prob_eco_) > 0:
        testing_data_new.loc[testing_data_new['Class_1'] == "Economic", "prob_class_2"] = predict_prob_eco_


# In[185]:


testing_data_new.drop_duplicates(keep="first", inplace=True)
testing_data_new.drop_duplicates(keep="first", inplace=True)
testing_data_new = testing_data_new.loc[testing_data_new['Class_1'] != "Default"]
testing_data_new = testing_data_new.loc[testing_data_new['Class_2'] != "Default"]
testing_data_new = testing_data_new.loc[testing_data_new['prob_class_1'] >= 0.65]
testing_data_new = testing_data_new.loc[testing_data_new['prob_class_2'] >= 0.25]


# In[170]:


if testing_data_new.empty == False:
    with open("co_st_data.txt", "rb") as myfile:
        dict_state_updated = pickle.load(myfile)

    with open("Con_data.txt", "rb") as myfile:
        a = pickle.load(myfile)
    africa_con = a[0][0]
    asia_con= a[0][1]
    europe_con= a[0][2]
    amer_con= a[0][3]
    oce_con= a[0][4]
    NA_con= a[0][5]

    testing_data_new['Feature'] = testing_data_new[["Headline","Summary"]].apply(lambda x: "".join(x),axis = 1)
    list_test_ = list(testing_data_new["Feature"])

    dict_f = {}
    for i in range(len(list_test_)):
        Country_ = set()
        Con_ = set()
        State_ = set()
        #print("News:-    ",str(i))
        biagram = ngrams(list_test_[i].split(' '),2)
        unigram = ngrams(list_test_[i].split(' '),1)
        trigram = ngrams(list_test_[i].split(' '),3)
        fourgram = ngrams(list_test_[i].split(' '),4)
        fivegram = ngrams(list_test_[i].split(' '),5)
        for uni,bia,tri,fr,five in zip(unigram,itertools.cycle(biagram),itertools.cycle(trigram),itertools.cycle(fourgram),itertools.cycle(fivegram)): 
            for country in dict_state_updated.keys():    
                uni = "".join(uni).split("'")[0]
                if (" ".join(bia).lower() == country.lower()) | ("".join(uni).lower() == country.lower()) | (" ".join(tri).lower() == country.lower()) | (" ".join(fr).lower() == country.lower()) | (" ".join(five).lower() == country.lower()):
                    Country_.add(country)
                    if country in africa_con:
                        Con_.add("Africa Con")
                    elif country in asia_con:
                        Con_.add("Asia Con")
                    elif country in europe_con:
                        Con_.add("Europe Con")
                    elif country in amer_con:
                        Con_.add("America Con")
                    elif country in oce_con:
                        Con_.add("Oceania Con")
                    elif country in NA_con:
                        Con_.add("North America Con")
            for ss1 in dict_state_updated.values():
                for s1 in ss1:
                    if ((" ".join(bia).lower() == str(s1).lower()) | ("".join(uni).lower() == str(s1).lower()) | (" ".join(tri).lower() == str(s1).lower()) | (" ".join(fr).lower() == str(s1).lower()) | (" ".join(five).lower() == str(s1).lower())):
                        if (s1.lower() == "south") | (s1.lower() == "east") | (s1.lower() == "west") | (s1.lower() == "north") | (s1.lower() == "road") | (s1.lower() == "drive") | (s1.lower() == "lot") | (s1.lower() == "providence") | (s1.lower() == "street") | (s1.lower() == "central") | (s1.lower() == "uc") | (s1.lower() == "male") | (s1.lower() == "field") | (s1.lower() == "green") | (s1.lower() == "worth")| (s1.lower() == "hill")| (s1.lower() == "island") | (s1.lower() == "acre") | (s1.lower() == "down") | (s1.lower() == "centre"):
                            pass
                        else:
                            State_.add(s1.lower())
        fin_loc = Country_.union(State_)
        dict_f[testing_data_new["Headline"].iloc[i]] = list(fin_loc)

    Fin_Dates = []
    for ind, tc in enumerate(list_test_):
        set1 = set()
        txt = set(tc.lower().split())
        remove_list = ["monday","tuesday","wednesday","thursday","friday","satursday","sunday"]
        days = set('Monday|Tuesday|Wednesday|Thursday|Friday|Satursday|Sunday'.lower().split('|')) & txt
        word_list = tc.lower().split(' ')
        sentens = ' '.join([i for i in word_list if i not in remove_list])
        matches = datefinder.find_dates(sentens)
        for match in matches:
            if str(match).startswith("20"):
                set1.add(str(match))
        fin_dates = days.union(set1)
        Fin_Dates.append(list(fin_dates))

    testing_data_new['Dates'] = Fin_Dates
    testing_data_new['Dates'] = testing_data_new['Dates'].astype(str).str[1:-1]
    del testing_data_new["Feature"]

    testing_data_new['Location'] = dict_f.values()
    testing_data_new['Location'] = testing_data_new['Location'].astype(str).str[1:-1]

    testing_data_new = splitDataFrameList(testing_data_new,"Location",",")
    testing_data_new.loc[testing_data_new["Location"] == ""] = np.nan
    testing_data_new.dropna(inplace=True)
    city_ = []
    lat_long_ = []
    for city in testing_data_new.Location.tolist():
        if "," not in city:
            lat_long_.append(get_lat_lng(city))    
            city_.append(city)
        else:
            for cities in city.split(","):
                city_.append(cities)
                lat_long_.append(get_lat_lng(cities))

    testing_data_new["Lat_Long"] = lat_long_
    testing_data_new["Lat_Long"] = testing_data_new['Lat_Long'].astype(str).str[1:-1]
    new = testing_data_new["Lat_Long"].str.split(",", n = 1, expand = True)
    testing_data_new["Latitude"]= new[0] 
    testing_data_new["Longitude"]= new[1] 
    del testing_data_new["Lat_Long"]
    testing_data_new.dropna(inplace=True)

    new_index= ['Source', 'Headline', 'url', 'Summary', 'Class_1', 'Class_2','Dates',"Location","Latitude","Longitude",'prob_class_1','prob_class_2']
    testing_data_new = testing_data_new[new_index]

    if testing_data_new[testing_data_new["Class_1"].isin(['Weather'])].empty == False:
        if os.path.isfile("Weather_output.xlsx"):
            testing_data_weather = pd.read_excel("Weather_output.xlsx")
            testing_data_new_weather = pd.concat([testing_data_new.loc[testing_data_new['Class_1'] == "Weather"],testing_data_weather]).reset_index(drop=True)
            testing_data_new_weather.drop_duplicates(["Headline","Location"],keep="first", inplace=True)
            testing_data_new_weather.to_excel("Weather_output.xlsx",index = False)
        else:
            testing_data_new.loc[testing_data_new['Class_1'] == "Weather"].to_excel("Weather_output.xlsx",index = False)

    if testing_data_new[testing_data_new["Class_1"].isin(['Economic'])].empty == False:
        if os.path.isfile("Economic_output.xlsx"):
            testing_data_economic = pd.read_excel("Economic_output.xlsx")
            testing_data_new_economic = pd.concat([testing_data_new.loc[testing_data_new['Class_1'] == "Economic"],testing_data_economic]).reset_index(drop=True)
            testing_data_new_economic.drop_duplicates(["Headline","Location"],keep="first", inplace=True)
            testing_data_new_economic.to_excel("Economic_output.xlsx",index = False)
        else:
            testing_data_new.loc[testing_data_new['Class_1'] == "Economic"].to_excel("Economic_output.xlsx",index = False)

    if testing_data_new[testing_data_new["Class_1"].isin(['Geopolitical'])].empty == False:
        if os.path.isfile("Geopolitical_output.xlsx"):
            testing_data_geopolitical = pd.read_excel("Geopolitical_output.xlsx",index = False)
            testing_data_new_geopolitical = pd.concat([testing_data_new.loc[testing_data_new['Class_1'] == "Geopolitical"],testing_data_geopolitical]).reset_index(drop=True)
            testing_data_new_geopolitical.drop_duplicates(["Headline","Location"],keep="first", inplace=True)
            testing_data_new_geopolitical.to_excel("Geopolitical_output.xlsx",index = False)
        else:
            testing_data_new.loc[testing_data_new['Class_1'] == "Geopolitical"].to_excel("Geopolitical_output.xlsx",index = False)

    if testing_data_new[testing_data_new["Class_1"].isin(['Disruptive'])].empty == False:
        if os.path.isfile("Disruptive_output.xlsx"):
            testing_data_disruptive = pd.read_excel("Disruptive_output.xlsx")
            testing_data_new_disruptive = pd.concat([testing_data_new.loc[testing_data_new['Class_1'] == "Disruptive"],testing_data_disruptive]).reset_index(drop=True)
            testing_data_new_disruptive.drop_duplicates(["Headline","Location"],keep="first", inplace=True)
            testing_data_new_disruptive.to_excel("Disruptive_output.xlsx",index = False)
        else:
            testing_data_new.loc[testing_data_new['Class_1'] == "Disruptive"].to_excel("Disruptive_output.xlsx",index = False)

    if os.path.isfile("All_data.xlsx"):
        testing_data_all = pd.read_excel("All_data.xlsx")
        testing_data_new_all = pd.concat([testing_data_new,testing_data_all]).reset_index(drop=True)
        testing_data_new_all.drop_duplicates(["Headline","Location"],keep="first", inplace=True)
        testing_data_new_all.to_excel("All_data.xlsx",index = False)
    else:
        testing_data_new.drop_duplicates(["Headline","Location"],keep="first", inplace=True)
        testing_data_new.to_excel("All_data.xlsx",index=False)
else:
    testing_data_new["Latitude"] = ""
    testing_data_new["Longitude"] = ""
    testing_data_new["Dates"] = ""
    testing_data_new["Location"] = ""
    new_index= ['Source', 'Headline', 'url', 'Summary', 'Class_1', 'Class_2','Dates',"Location","Latitude","Longitude",'prob_class_1','prob_class_2']
    testing_data_new = testing_data_new[new_index]
    if os.path.isfile("All_data.xlsx"):
        testing_data_all = pd.read_excel("All_data.xlsx")
        testing_data_new_all = pd.concat([testing_data_new,testing_data_all]).reset_index(drop=True)
        testing_data_new_all.drop_duplicates(["Headline","Location"],keep="first", inplace=True)
        testing_data_new_all.to_excel("All_data.xlsx",index = False)
    else:
        testing_data_new.drop_duplicates(["Headline","Location"],keep="first", inplace=True)
        testing_data_new.to_excel("All_data.xlsx",index=False)

