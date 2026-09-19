import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv(r"C:\Users\SV\Desktop\CODE\Second Project\spam.csv")
data.drop_duplicates(inplace=True)
data['spam'] = data['spam'].replace([0,1],['Not Spam', 'Spam'])
text = data['text']
sp = data['spam']
(text_train, text_test, sp_train, sp_test) = train_test_split(
    text,sp, test_size=0.2
)
cv = CountVectorizer(stop_words = 'english')
features = cv.fit_transform(text_train)
#creating model
model = MultinomialNB()
model.fit(features, sp_train)
#test
features_test = cv.transform(text_test)
#predict
def predict(sample):
 sample = cv.transform([sample]).toarray()
 result = model.predict(sample)
 return result
input_1 = input("Give the mail :")
output = predict(input_1)
print(output)