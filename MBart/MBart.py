import streamlit as st
import numpy as np
import pickle as pkl

def translate(text, tokenizer, model, target_lang):
    input_ids = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate( **input_ids, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_text

def main():
    st.title("MBart Translation App")
    text = st.text_area("Enter the text:")
    target_lang = st.selectbox("Select target language:", ["hi_IN", "ta_IN", "te_IN", "ml_IN", "mr_IN"])
    with open('model.pkl', 'rb') as handle:
        model = pkl.load(handle)
        
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pkl.load(handle)

    if st.button("Translate"):
        answer = translate(text, tokenizer, model, target_lang)
        answer_display = st.empty()
        answer_display.text_area("Translated Text:", value=answer, height=10)

if __name__ == "__main__":
    main()