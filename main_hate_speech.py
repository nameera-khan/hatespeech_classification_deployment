import streamlit as st
import torch
import torch.nn as nn
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
import numpy as np
import pandas as pd
import time

@st.cache(allow_output_mutation=True)
def get_model(): 
  tokenizer = tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
  model = AutoModel.from_pretrained("decodedplot/hatespeech-roberta")
  return tokenizer, model

tokenizer,model = get_model()

user_input = st.text_area("Enter Text To Detect Hate")
button = st.button("Analyse")

d = { 1:'Hate',
     0:'Not Hate'}

if user_input and button: 
  test_sample = tokenizer([user_input], padding=True, truncation=True, return_tensors='pt')
  output = model(**test_sample)
  st.write("Logits: ", output.logits)
  y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
  st.write("Detected: ", d[y_pred[0]])


