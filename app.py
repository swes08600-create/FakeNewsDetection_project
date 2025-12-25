import streamlit as st
import pickle

model = pickle.load(open("fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("📰 Fake News Detection App")

news_text = st.text_area("Enter News Text")

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter news text")
    else:
        vector = tfidf.transform([news_text])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")