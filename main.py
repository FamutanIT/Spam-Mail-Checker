import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#read the file
data1 = pd.read_csv(r"C:\Users\SV\Desktop\CODE\Spam checker\data\spam.csv")
data2 = pd.read_csv(r"C:\Users\SV\Desktop\CODE\Spam checker\data\mail_data.csv")
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
#test
#features_test = cv.transform(text_test)
#print(model.score(features_test, sp_test))
#predict
def predict(sample):
 sample = cv.transform([sample]).toarray()
 result = model.predict(sample)
 return result
input_1 = input("Give the mail :")
output = predict(input_1)
print(output)