import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

import joblib

# load dataset
df = pd.read_csv('./data/customer_messages_dataset.csv')
# print(df.head())

# split into X and y
X = df["message"]
y = df["label"]

# encode y
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Feature Extraction
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# create an instance of the model
model = SVC()

# fit the data
model.fit(X_train, y_train)

# evaluate the model
print(model.score(X_test, y_test))
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# print(cm)

# save model, vectorizer and encoder
# joblib.dump(model, "./models/task_classifier.pkl")
# joblib.dump(vectorizer, "./models/vectorizer.pkl")
# joblib.dump(encoder, "./models/encoder.pkl")