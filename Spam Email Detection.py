#1
import numpy as np
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import re

#2
data = pd.read_csv("/content/emails.csv")
X = data['email']
Y= data['label']

def clean_email(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

X_cleaned = X.apply(clean_email)
X_cleaned = X_cleaned.str.replace(r'\bsubject\b', ' ', regex=True) 
data['email'] = X_cleaned
columns_to_keep = ['email', 'label']
data = data[columns_to_keep]
X= data ['email']
Y= data['label']
print(data)

#3
vectorizer = TfidfVectorizer(max_features=1000)  
X_features = vectorizer.fit_transform(X).toarray()
X_features= pd.DataFrame(X_features)

#4
X_train, X_test, y_train, y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42)

#5
model = Sequential()
model.add(Dense(128, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  

#6
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

#7
y_train = pd.to_numeric(y_train, errors='coerce') 
y_train = y_train.fillna(0).astype(int) 
y_test = pd.to_numeric(y_test, errors='coerce')
y_test = y_test.fillna(0).astype(int)

#8
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

#9
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))