import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from pathlib import Path
#read the file
BASE_DIR = Path(__file__).resolve().parent
data1p = BASE_DIR / "data" / "spam.csv"
data2p = BASE_DIR / "data" / "mail_data.csv"
data1 = pd.read_csv(data1p)
data2 = pd.read_csv(data2p)
#combine thease file each file
data1['spam'] = data1['spam'].replace([0,1] ,['Not Spam', 'Spam',])
data2['spam'] = data2['Category'].replace(['ham','spam'], ['Not Spam', 'Spam'])
data2.drop(columns=['Category'], inplace= True)
data2.rename(columns={'Message':'text'}, inplace=True)
data = pd.concat([data1,data2], ignore_index=True)
data.drop_duplicates(inplace=True)
text = data['text']
sp = data['spam']
(text_train, text_test, sp_train, sp_test) = train_test_split(
    text,sp, test_size=0.2,
)
cv = CountVectorizer(stop_words = 'english')
features = cv.fit_transform(text_train)
#creating model
model = MultinomialNB()
model.fit(features, sp_train)
#features_test = cv.transform(text_test)       #test
#print(model.score(features_test, sp_test))
#predict
def predict(sample):                   #This is the function to predict
 sample = cv.transform([sample]).toarray()
 result = model.predict(sample)
 return result
input_1 = input("Give the mail :")    #This is the example if you want do it in the bash
output = predict(input_1)
print(output)
#st.header("THIS IS A EMAIL SPAM CHECKER", divider="red")   # this is the streamlit web
#email = st.text_input("Enter your mail need to check here:")
#output = predict(email)
#st.title("Here your answers:")
#st.text(output)
