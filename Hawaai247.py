import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from textblob import TextBlob
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import xlrd
import time

driver = webdriver.PhantomJS("C://Users//vishrut.b//PycharmProjects//Toshiba//phantomjs-2.1.1-windows//bin//phantomjs.exe")
USER_AGENT = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

wordnet_lemmatizer = WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()
head_link_df = pd.DataFrame()
output_df = pd.DataFrame()


base_url = 'https://www.hawaii247.com/page/'
link_excel_name = 'Hawaai247Links.xlsx'
data_excel_name = 'Hawaai247Data.xlsx'


def get_only_text(url):
    driver.get(url)
    time.sleep(30)
    page = driver.page_source.encode('utf-8')
    soup = BeautifulSoup(page, features='lxml')
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return text

#
# # Get Headlines and Links
#
# x = 1
#
# while x < 3:
#
#     url = base_url+str(x)
#     driver.get(url)
#     page = driver.page_source.encode('utf-8')
#     soup = BeautifulSoup(page, features='lxml')
#     articles = soup.find_all('div', {'class': 'box-post-content'})
#     for article in articles:
#         anchor = article.find('a')
#         link = anchor['href']
#         headline = anchor['title']
#         head_link_df = head_link_df.append({'Headline': headline, 'URL': link}, ignore_index=True)
#     x += 1
#
# headwriter = pd.ExcelWriter(link_excel_name)
# head_link_df.to_excel(excel_writer=headwriter, sheet_name='Sheet1')
# headwriter.save()
# headwriter.close()

# Get Text, Summary, Sentiment
book = xlrd.open_workbook(link_excel_name)
first_sheet = book.sheet_by_index(0)
numrows = first_sheet.nrows

x = 1

while x < numrows:

    headline = first_sheet.cell_value(rowx=x, colx=1)
    link = first_sheet.cell_value(rowx=x, colx=2)

    try:
        textdata = get_only_text(link)
        words = sent_tokenize(textdata)
        freqTable = dict()
        words = [word.lower() for word in words]
        # words = [word for word in words if word.isalnum()]
        words = [wordnet_lemmatizer.lemmatize(word) for word in words]
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english')
        tfidf_matrix = tf.fit_transform(words)
        idf = tf.idf_
        dict_idf = dict(zip(tf.get_feature_names(), idf))
        sentences = sent_tokenize(textdata)
        sent_list = set()
        for sentence in sentences:
            for value, term in dict_idf.items():
                if value in sentence:
                    sent_list.add(sentence)

        summary = summarize("\n".join(sent_list), ratio=0.1)
        scores = sid.polarity_scores("\n".join(sent_list))
        sentiment_score = scores['compound']

        output_df = output_df.append({'Headline': headline, 'URL': link,
                                      'Summary': summary, 'Sentiment': sentiment_score}, ignore_index=True)
        x += 1
        print('Scraped at:' + link)
    except:
        print('Could not find text at:' + link)
        x += 1

data_writer = pd.ExcelWriter(data_excel_name)
output_df.to_excel(data_writer, sheet_name='Sheet1')
data_writer.save()
data_writer.close()

