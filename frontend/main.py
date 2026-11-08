import os
import requests
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Cenayang Tulisan Tangan", page_icon="", layout="centered")

st.markdown(
    """
    <style>
    /* Hero Section */
    .hero {
        background: #F5CB63;
        color: #49391F;
        padding: 1.8rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #49391F;
        margin: 0;
        font-size: 2rem;
    }

    .hero p {
        color: #665333;
        margin: .45rem 0 0;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #EADFC7;
        color: #49391F;
        padding: 1rem;
        border-radius: 14px;
    }

    /* Primary Button */
    div.stButton > button[kind="primary"] {
        background: #E9B94F;
        border: 1px solid #D6A83F;
        color: #49391F;
        border-radius: 10px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #F5CB63;
        border-color: #D6A83F;
        color: #49391F;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #D6B96E;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Custom Notification */
    .notice {
        background: #FFF1CC;
        border: 1px solid #E9B94F;
        color: #49391F;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }

    .notice.success {
        background: #F5ECD8;
        border-color: #D6B96E;
    }
    </style>

    <div class="hero">
        <h1>Klasifikasi Tulisan Tangan</h1>
        <p>Unggah citra tulisan tangan untuk mengenali nomor identitas</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Backend Configuration
api_url = os.getenv(
    "PREDICT_URL",
    "http://127.0.0.1:8000/predict"
)

# Developer Settings
with st.expander("Pengaturan Pengembang"):
    st.caption("Koneksi Backend")

    health_url = api_url.rsplit("/", 1)[0] + "/health"

    if st.button("Periksa Koneksi"):
        try:
            health_response = requests.get(
                health_url,
                timeout=5,
            )
            health_response.raise_for_status()

            st.success("Backend berhasil terhubung!")

        except requests.RequestException as error:
            st.error(f"Backend gagal terhubung: {error}")

st.write("Pilih citra tulisan tangan yang jelas untuk mendapatkan hasil prediksi terbaik!")
uploaded_file = st.file_uploader("Unggah Citra", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Citra yang akan dikenali", use_container_width=True)

    if st.button("Kenali Tulisan Tangan", type="primary", disabled=uploaded_file is None):
        try:
            with st.spinner("Model sedang memproses citra dan memprediksi kelas..."):
                response = requests.post(
                    api_url,
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()

            label = result["Label"]
            probabilities = result["Probability"]

            predicted_id = class_names[label]
            confidence = float(probabilities[label])
            class_labels = [str(i) for i in range(len(probabilities))]

            st.markdown(
                '<div class="notice success">Prediksi Berhasil!</div>',
                unsafe_allow_html=True,
            )

            left, right = st.columns(2)
            left.metric("Prediksi Identitas:", str(label))
            right.metric("Tingkat Keyakinan", f"{confidence:.1%}")

            st.subheader("Probabilitas Setiap Kelas")
            st.caption(
                "Nilai probabilitas diperoleh langsung dari hasil prediksi model"
            )

            # Prepare probability data
            class_labels = class_names

            probability_df = pd.DataFrame({
                "Kelas": class_labels,
                "Probabilitas": [
                    float(value) for value in probabilities
                ],
            })

            # Create horizontal bar chart
            chart = (
                alt.Chart(probability_df)
                .mark_bar(
                    color="#E9B94F",
                    cornerRadiusEnd=5,
                )
                .encode(
                    x=alt.X(
                        "Probabilitas:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(
                            format="%",
                            title="Probabilitas",
                            tickCount=6,
                        ),
                    ),
                    y=alt.Y(
                        "Kelas:N",
                        sort=class_labels,
                        axis=alt.Axis(
                            title="Kelas",
                            labelOverlap=False,
                            labelFontSize=12
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Class:N"),
                        alt.Tooltip(
                            "Probabilitas:Q",
                            format=".2%",
                        ),
                    ],
                )
                .properties(
                    height=max(300, len(class_labels) * 30)
                )
            )

            st.altair_chart(
                chart,
                use_container_width=True,
            )

        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            st.error(f"Prediksi gagal. Periksa backend sudah berjalan dan respon sesuai format. Detail: {error}")
else:
    st.markdown(
        '<div class="notice">'
            'Unggah Citra PNG atau JPG untuk memulai ^.^'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

st.markdown(
    """
    <div style="text-align: center; padding: 0.5rem 0;">
        <p style="color: #806F50; font-size: 0.85rem;">
            Cenayang Tulisan Tangan
            <br>
            Klasifikasi Tulisan Tangan dengan Vision Transformer
        </p>
        <p style="color: #A39478; font-size: 0.75rem;">
            Dikembangkan dengan Mata Batin oleh Kelompok A5
            <br>
            © 2026 All rights reserved.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)