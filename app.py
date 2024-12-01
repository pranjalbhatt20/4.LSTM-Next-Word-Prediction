import streamlit as st
import numpy as np 
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load LSTM model
model=load_model('./Saved Model/model.h5')

# load tokenizer 
with open('./Saved Model/tokenizer.pkl','rb') as f:
    tokenizer=pickle.load(f)

## Predicting the next word 
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):] # Ensure sequence len matches max Sequence 
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)[0]
    print(predicted_word_index)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

# Streamlit App
st.title("How would shakespere Write ")
input_text=st.text_input("Enter Sequence of words")
if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len=max_sequence_len)
    st.write(f"Next Word :{next_word}")
