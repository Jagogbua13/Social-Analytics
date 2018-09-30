import tweepy
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from config import (consumer_key,
                    consumer_secret,
                    access_token,
                    access_token_secret)
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
from pprint import pprint
import matplotlib.patches as mpatches

# setting targets
target_users = ("@BBCWorld", "@CBS", "@CNN", "@FoxNews","@nytimes")
color = ["blue","green","yellow","purple","red"]
now = datetime.datetime.now()
now2 = now.strftime("%m/%d/%y")
sentiment_list = []
text_list = []

#creating a loop to go through targets
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
tweet_text = []
target_list = []
for target,colors in zip(target_users,color) :
    
    tweet_count = 1
    
   
    #creating loop to go through pages of tweets, there are 20 tweets per page and we loop through 5
    for x in range(1,6) :
        public_tweets = api.user_timeline(target,page = x)
        
        
        # looping through each tweet's text and analyzing them
        for tweet in public_tweets :
            results = analyzer.polarity_scores(tweet['text'])
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]
            text= tweet['text']
            #appending list
            compound_list.append(compound)            
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
            tweet_text.append(text)
            target_list.append(target)

            tweet_count = tweet_count + 1
           
        #appending my list for scatter plots
            sentiment_text = {
            "User": target_list,
            "Compound": compound_list,
            "Positive": positive_list,
            "Neutral": negative_list,
            "Negative": neutral_list,
            "Text":tweet_text
            }
        
            # appending list for bar plots- this is also unnessary and i could just use the dfs i use later
    sentiment = {
    "User": target,
    "Compound": np.mean(compound_list),
    "Positive": np.mean(positive_list),
    "Neutral": np.mean(negative_list),
    "Negative": np.mean(neutral_list)
        
    }
    sentiment_list.append(sentiment)

    test_df= pd.DataFrame(sentiment_text)
test_df.to_csv("analysis/" + News_tweet_sentiment_data, encoding="utf-8")   

#creating dfs for each twitter handle and adding a tweet counter
initial_value = 1
bbc_df = test_df[test_df['User'] == "@BBCWorld"]
cbs_df = test_df[test_df['User'] == "@CBS"]
cnn_df = test_df[test_df['User'] == "@CNN"]
fox_df = test_df[test_df['User'] == "@FoxNews"]
ny_df = test_df[test_df['User'] == "@nytimes"]
ny_df['counter'] = range(initial_value, len(ny_df) +initial_value)
fox_df['counter'] = range(initial_value, len(fox_df) +initial_value)
cnn_df['counter'] = range(initial_value, len(cnn_df) +initial_value)
cbs_df['counter'] = range(initial_value, len(cbs_df) +initial_value)
bbc_df['counter'] = range(initial_value, len(bbc_df) +initial_value)
test_df.head()
ny_df.head()

#plotting  my sentiments for each news outlet
plt.scatter(y=fox_df['Compound'],x=fox_df['counter'],c="blue",label='Fox',edgecolor="black",marker="o")
plt.scatter(y=ny_df['Compound'],x=ny_df['counter'],c="green",label='NY Times',edgecolor="black",marker="o")
plt.scatter(y=cnn_df['Compound'],x=cnn_df['counter'],c="yellow",label='CNN',edgecolor="black",marker="o")
plt.scatter(y=cbs_df['Compound'],x=cbs_df['counter'],c="purple",label='CBS',edgecolor="black",marker="o")
plt.scatter(y=bbc_df['Compound'],x=bbc_df['counter'],c="red",label='BBC',edgecolor="black",marker="o")
#making the graph look a bit better
plt.title(f"Sentiment Analysis of Media tweets ({now2})")
plt.xlabel("Tweet count")
plt.ylabel("Tweet polarity")
plt.ylim(1, -1)
plt.xlim(0, 100)
plt.grid(True)
plt.margins(0,0)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.savefig("analysis/Fig1.png")

news_df = pd.DataFrame(sentiment_list)#.round(2)
print(bbc_df['Compound'].mean())
news_df.head()

plt.bar(x= news_df["User"],height= news_df["Compound"],color=("blue","green","yellow","purple","red"))
plt.title(f"Overall Media Sentiment Based on twitter ({now2})")
plt.ylabel("Tweet polarity")
plt.savefig("analysis/Fig2.png")