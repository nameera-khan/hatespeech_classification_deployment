import streamlit as st
import torch
import torch.nn as nn
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
import numpy as np
import pandas as pd
import time

st.header("Understanding Model Performance")
st.markdown("---")

st.markdown("""
### 🔬 About This Model & Research

This demo showcases a model trained for hate speech classification across multiple platforms (Twitter, Reddit, Gab) - a task known for its complexity and subjectivity.

**Key Challenges:**
- Hate speech is highly contextual and platform-specific
- Models often face precision-recall trade-offs
- Generalization across diverse online communities remains difficult

**Design Approach:**
This demo is optimized to flag *potential* hate speech for **human review**, prioritizing recall to ensure comprehensive coverage, even if it means some non-harmful content is also flagged.

**Real-World Application:**
In production, this would serve as a first-pass filter, with predictions undergoing human moderator review before any action is taken.

---

### 📊 Research Findings: The Cross-Platform Generalization Challenge

Our analysis revealed significant performance variations:

- **Twitter Data:** - Strong performance on mainstream platform
- **Gab-Reddit Combined:** Good performance - Benefits from cross-platform training
- **Reddit Alone:** Lower performance - Reveals unique linguistic patterns

This highlights why one-size-fits-all content moderation often fails and why platform-specific approaches may be necessary.
""")


@st.cache(allow_output_mutation=True)
def get_model(): 
  tokenizer = tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
  model = AutoModelForSequenceClassification.from_pretrained("decodedplot/hatespeech-roberta")
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


