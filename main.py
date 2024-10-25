import numpy as np
import tensorflow as tf 
from  tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import streamlit as st

## load the imdb dataset word index 
word_index=imdb.get_word_index()
revese_word_index = {value:key for key , value in word_index.items()}

## load the pre-trained model with ReLu activation 
model = load_model('simple_rnn_imdb.h5')






# Step2 : Helper Functions 
#Funtion to decoder reviews 4
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review ])

## Function  to preprocess user input 
def preprocess_text(text): 
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words ]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


## stap 3 : prediction function 
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction= model.predict(preprocessed_input)

    sentiment = 'positive' if prediction [0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]

## streamet app 

st.title('IMDB Movie review Sentiment analisis ')
st.write('Enter a moview to classify it as positive or negative .')

## user input 
user_input = st.text_area('Movie Reviw')

if st.button('Classify'):
    preprocessd_input = preprocess_text(user_input)

    ## make prediction 
    prediction = model.predict(preprocessd_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    ## Display the result 
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction Score :{prediction[0][0]}')
else:
    st.write('Plese enter a movie review ')