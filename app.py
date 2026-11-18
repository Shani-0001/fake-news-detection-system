
import streamlit as st
import joblib
import re

# 1. Configure website
st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="📰",
    layout="centered"
)

# 2. Load saved machine learning model
MODEL_PATH = "model/fake_news_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# 3. Text preprocessing
def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# 4. Website heading
st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below to analyze "
    "whether it resembles fake or genuine news "
    "in the training dataset."
)

st.divider()

# 5. User input
news = st.text_area(
    "Enter your news article:",
    height=220,
    placeholder="Paste your news article here..."
)

# 6. Analyze button
if st.button("Analyze News", type="primary"):

    if not news.strip():

        st.warning("Please enter a news article.")

    elif len(news.strip()) < 30:

        st.warning("Please enter a longer news article.")

    else:

        # Clean input
        cleaned_news = clean_text(news)

        if not cleaned_news:

            st.warning("No usable text found.")

        else:

            # Make prediction
            prediction = model.predict([cleaned_news])[0]

            # Get prediction probabilities
            probabilities = model.predict_proba(
                [cleaned_news]
            )[0]

            classes = list(model.classes_)

            predicted_score = probabilities[
                classes.index(prediction)
            ] * 100

            st.divider()

            # 7. Display prediction
            if predicted_score < 60:

                st.warning(
                    "⚠️ RESULT: NEEDS VERIFICATION"
                )

            elif prediction == 0:

                st.error(
                    "❌ PREDICTION: FAKE NEWS"
                )

            else:

                st.success(
                    "✅ PREDICTION: GENUINE NEWS"
                )

            # 8. Display confidence score
            st.metric(
                "Model confidence score",
                f"{predicted_score:.2f}%"
            )

            st.progress(
                int(predicted_score)
            )

            # 9. Disclaimer
            st.info(
                "This prediction reflects learned patterns "
                "in the training dataset. It does not verify "
                "whether the news claim is factually true."
            )
