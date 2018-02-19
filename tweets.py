## import the libraries
import tweepy, codecs
from textblob import TextBlob
import csv,io
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

## fill in your Twitter credentials 
consumer_key = "XYhJIKHoOjqSQlN6lKHGUTPyz"
consumer_secret = "2oBPWminOaWChuLU9T4eFcW5SE5nj0wgKTD6NSWRY8TGM6YKbL"
access_token = "55470209-TwVM0w2DvQYxKUFsWsRhBD59ffEEpFxTkPpsbHM1h"
access_token_secret = "y62sBAglIptaIQvd0293KGMtnE3IJGNFVwcfGemUhUymX"

## let Tweepy set up an instance of the REST API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

## fill in your search query and store your results in a variable
results = api.search(q = "BABA", lang = "en", result_type = "recent", count = 3000)
##print(results)

## use the codecs library to write the text of the Tweets to a .txt file
file = codecs.open("google_tweet.txt", "w", "utf-8")
with io.open('sentiment.csv' ,'w' , encoding='utf8' , newline='') as csvfile:
	csv_writer = csv.writer(csvfile)
	csv_writer.writerow(["Tweet","Sentimentvalue","Sentiment"])
	for result in results:
		file.write(result.text)
		file.write("\n")
		analysis=TextBlob(result.text)
		sentiment_val = analysis.sentiment.polarity
		print(analysis,sentiment_val)
		if sentiment_val > 0:
			output =  'positive'
		elif sentiment_val == 0 :
			output = 'neutral' 
		else: 
			output = 'negative' 
		csv_writer.writerow([result.text,sentiment_val,output])
	
file.close()

with io.open ('sentiment.csv' , 'r' , encoding='utf-8' ) as csvfile:
	df = pd.read_csv(csvfile)
	sent=df["Sentiment"]
	counter=Counter(sent)
	positive = counter['positive']
	negative=counter['negative']
	neutral=counter['neutral']

labels = 'Positive' , 'Negative' , 'Neutral' 
sizes = [positive,negative,neutral]
colors = ['green', 'red','grey']
yourtext= "apple"


plt.pie(sizes,labels = labels ,colors = colors , shadow = True ,startangle=90)
plt.title("Sentiment of tweets contining the word " + yourtext)
plt.show()