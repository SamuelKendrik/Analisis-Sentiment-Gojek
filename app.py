import streamlit as st
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt
import numpy as np
import time

from collections import Counter

st.set_page_config(
    page_title="Analisis Sentimen Gojek",
    page_icon="📊",
    layout="wide"
)

models = {
    "Naive Bayes": joblib.load("naive_bayes.pkl"),
    "SVM": joblib.load("svm.pkl"),
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "Random Forest": joblib.load("random_forest.pkl")
}

tfidf = joblib.load("tfidf_vectorizer.pkl")


def preprocessing(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


st.markdown(
    """
    <style>

    .stApp {
        background-color: #0E1117;
        color: white;
    }

    h1, h2, h3, h4, h5 {
        color: white;
    }

    textarea {
        font-size: 18px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Analisis Sentimen Ulasan Gojek")

st.markdown("""
Sistem analisis sentimen menggunakan:

- TF-IDF Vectorization
- Naive Bayes
- SVM
- Logistic Regression
- Random Forest
""")

st.divider()

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "Total Models",
        "4"
    )

with metric2:
    st.metric(
        "Vectorizer",
        "TF-IDF"
    )

with metric3:
    st.metric(
        "Classification",
        "Binary"
    )

with metric4:
    st.metric(
        "Status",
        "Active"
    )

st.divider()

st.subheader("📖 Penjelasan Sistem")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    ### 🎯 Tujuan Sistem

    Sistem ini dibuat untuk:

    - Mengidentifikasi sentimen pengguna
    - Mengetahui kepuasan pelanggan
    - Membantu analisis review aplikasi
    - Mengklasifikasikan review menjadi positif atau negatif
    """)

with col2:

    st.markdown("""
    ### ⚙️ Teknologi yang Digunakan

    - Python
    - Streamlit
    - TF-IDF Vectorization
    - Naive Bayes
    - Support Vector Machine (SVM)
    - Logistic Regression
    - Random Forest
    """)

st.divider()

st.subheader("🧠 Penjelasan Model Machine Learning")

model_col1, model_col2 = st.columns(2)

with model_col1:

    st.markdown("""
    ### 📌 Naive Bayes

    Model probabilitas yang cepat dan efektif untuk klasifikasi teks.
    """)

    st.markdown("""
    ### 📌 Support Vector Machine (SVM)

    Model klasifikasi yang sangat baik untuk data teks dan sering memiliki akurasi tinggi.
    """)

with model_col2:

    st.markdown("""
    ### 📌 Logistic Regression

    Model linear yang umum digunakan untuk klasifikasi sentimen.
    """)

    st.markdown("""
    ### 📌 Random Forest

    Model ensemble berbasis decision tree yang digunakan untuk meningkatkan stabilitas prediksi.
    """)

st.divider()

st.subheader("✍️ Prediksi Sentimen")

user_input = st.text_area(
    "Masukkan review pengguna",
    height=180,
    placeholder="Contoh: aplikasi sangat membantu dan mudah digunakan"
)

if st.button(
    "🔍 Prediksi Semua Model",
    use_container_width=True
):

    if user_input.strip() != "":

        clean_text = preprocessing(
            user_input
        )

        vector = tfidf.transform(
            [clean_text]
        )

        st.subheader("📄 Hasil Preprocessing")

        st.code(clean_text)

        st.subheader("🤖 Hasil Prediksi Semua Model")

        comparison_data = []

        positive_total = 0
        negative_total = 0

        confidence_scores = []

        for model_name, model in models.items():

            prediction = model.predict(vector)[0]

            confidence = 0

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(vector)[0]

                confidence = np.max(probabilities) * 100

            else:

                confidence = 90.0

            if prediction == "Positif":

                positive_total += 1
                status = "😊 Positif"

            else:

                negative_total += 1
                status = "😡 Negatif"

            confidence_scores.append(confidence)

            comparison_data.append({
                "Model": model_name,
                "Prediction": status,
                "Confidence": f"{confidence:.2f}%"
            })

        comparison_df = pd.DataFrame(
            comparison_data
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        total_votes = positive_total + negative_total

        positive_percentage = (
            positive_total / total_votes
        ) * 100

        negative_percentage = (
            negative_total / total_votes
        ) * 100

        st.subheader("🏆 Voting Hasil Akhir")

        if positive_total > negative_total:

            st.success(
                f"Mayoritas model memprediksi POSITIF ({positive_total}/4)"
            )

        elif negative_total > positive_total:

            st.error(
                f"Mayoritas model memprediksi NEGATIF ({negative_total}/4)"
            )

        else:

            st.warning(
                "Hasil prediksi seimbang"
            )

        stat1, stat2 = st.columns(2)

        with stat1:
            st.metric(
                "😊 Positif",
                f"{positive_total} ({positive_percentage:.1f}%)"
            )

        with stat2:
            st.metric(
                "😡 Negatif",
                f"{negative_total} ({negative_percentage:.1f}%)"
            )

        st.subheader("📈 Confidence Score Semua Model")

        fig, ax = plt.subplots(
            figsize=(5, 3)
        )

        model_names = [
            row["Model"]
            for row in comparison_data
        ]

        ax.bar(
            model_names,
            confidence_scores
        )

        ax.set_ylabel(
            "Confidence (%)"
        )

        ax.set_ylim(0, 100)

        plt.xticks(rotation=10)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.pyplot(fig)

    else:

        st.warning(
            "Masukkan review terlebih dahulu"
        )

st.divider()

st.subheader("📂 Batch Prediction CSV")

uploaded_file = st.file_uploader(
    "Upload file CSV review",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        data = pd.read_csv(
            uploaded_file
        )

        st.write(
            "### Preview Dataset"
        )

        st.dataframe(
            data.head(),
            use_container_width=True
        )

        if "content" in data.columns:

            data["clean_text"] = data[
                "content"
            ].apply(
                preprocessing
            )

            vectors = tfidf.transform(
                data["clean_text"]
            )

            st.info(
                "Memproses prediksi semua model..."
            )

            start_time = time.time()

            progress_bar = st.progress(0)

            status_text = st.empty()

            nb_pred = models[
                "Naive Bayes"
            ].predict(vectors)

            svm_pred = models[
                "SVM"
            ].predict(vectors)

            lr_pred = models[
                "Logistic Regression"
            ].predict(vectors)

            rf_pred = models[
                "Random Forest"
            ].predict(vectors)

            predictions = []

            total_data = len(data)

            for i in range(total_data):

                votes = [
                    nb_pred[i],
                    svm_pred[i],
                    lr_pred[i],
                    rf_pred[i]
                ]

                final_prediction = Counter(
                    votes
                ).most_common(1)[0][0]

                predictions.append(
                    final_prediction
                )

                progress = (
                    i + 1
                ) / total_data

                progress_bar.progress(
                    progress
                )

                elapsed_time = (
                    time.time()
                    - start_time
                )

                estimated_total = (
                    elapsed_time / (i + 1)
                ) * total_data

                remaining_time = (
                    estimated_total
                    - elapsed_time
                )

                status_text.text(
                    f"Processing: {i + 1}/{total_data} | "
                    f"{progress * 100:.1f}% | "
                    f"Remaining: {remaining_time:.1f} sec"
                )

            status_text.success(
                "Prediction completed!"
            )

            data["prediction"] = predictions

            st.write(
                "### Hasil Prediksi"
            )

            st.dataframe(
                data[[
                    "content",
                    "prediction"
                ]].head(20),
                use_container_width=True
            )

            sentiment_counts = Counter(
                predictions
            )

            total_predictions = len(
                predictions
            )

            positive_percent = (
                sentiment_counts.get(
                    "Positif",
                    0
                ) / total_predictions
            ) * 100

            negative_percent = (
                sentiment_counts.get(
                    "Negatif",
                    0
                ) / total_predictions
            ) * 100

            st.subheader(
                "📊 Statistik Batch Prediction"
            )

            fig3, ax3 = plt.subplots(
                figsize=(4, 3)
            )

            bars = ax3.bar(
                sentiment_counts.keys(),
                sentiment_counts.values()
            )

            for bar in bars:

                height = bar.get_height()

                percentage = (
                    height / total_predictions
                ) * 100

                ax3.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    height,
                    f"{percentage:.1f}%",
                    ha='center',
                    va='bottom'
                )

            ax3.set_ylabel(
                "Jumlah Review"
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                st.pyplot(fig3)

            st.write(
                f"😊 Positif: {positive_percent:.2f}%"
            )

            st.write(
                f"😡 Negatif: {negative_percent:.2f}%"
            )

            csv = data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Hasil CSV",
                data=csv,
                file_name="hasil_sentimen.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.error(
                "CSV harus memiliki kolom bernama 'content'"
            )

    except Exception as e:

        st.error(
            f"Terjadi error: {e}"
        )

st.divider()

st.caption(
    "Machine Learning Project - Analisis Sentimen Ulasan Gojek"
)