import streamlit as st
import torch
import numpy as np
import pickle as pkl

def translate(text, tokenizer, model):
    input_ids = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate( **input_ids, forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"])
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_text

def main():
    st.title("MBart Translation App")
    text = st.text_area("Enter the text:")
    with open('model.pkl', 'rb') as handle:
        model = pkl.load(handle)
        
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pkl.load(handle)

    if st.button("Answer"):
        answer = translate(text, tokenizer, model)
        answer_display = st.empty()
        answer_display.text_area("Answer:", value=answer, height=10)

if __name__ == "__main__":
    main()