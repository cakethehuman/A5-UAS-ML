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
        <h1>Klasifikasi Tulisan Tangan berbasis DINOv2</h1>
        <p>Unggah citra tulisan tangan untuk mengenali identitas pemilik tulisan tangan!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Backend Configuration
api_url = os.getenv(
    "PREDICT_URL",
    "http://127.0.0.1:8000/predict"
)

# Load NIM Mapping
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(BASE_DIR, "nim_nama.txt")

class_mapping = {}

with open(mapping_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(maxsplit=2)

        if len(parts) == 3:
            index, kode_nim, nama = parts

            class_mapping[int(index)] = {
                "kode_nim": kode_nim,
                "nama": nama
            }

# Class Labels
class_names = [
    f'{data["kode_nim"]} - {data["nama"]}'
    for index, data in sorted(class_mapping.items())
]

# Backend Connection
if st.button("Periksa Koneksi Backend"):
    try:
        health_response = requests.get(
            api_url.rsplit("/", 1)[0] + "/health",
            timeout=5,
        )
        health_response.raise_for_status()

        st.success("Koneksi berhasil!")

    except requests.RequestException:
        st.error("Backend gagal terhubung.")

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

                predicted_index = int(result["Label"])

                probabilities = result["Probability"]

                if probabilities and isinstance(probabilities[0], list):
                    probabilities = probabilities[0]

                confidence = float(probabilities[predicted_index])

                if predicted_index == 51:
                    st.warning(
                        "Tulisan tidak dikenali. "
                        "Silakan unggah citra yang lebih jelas."
                    )

                else:
                    predicted_student = class_mapping.get(predicted_index)

                    if predicted_student:
                        label = (
                            f'{predicted_student["kode_nim"]} - '
                            f'{predicted_student["nama"]}'
                        )
                    else:
                        label = f"Kelas {predicted_index}"

                    st.success("Prediksi berhasil!")

                    left, right = st.columns(2)

                    left.metric("Prediksi Identitas", label)
                    right.metric("Tingkat Keyakinan", f"{confidence:.1%}")
                    
            st.markdown(
                '<div class="notice success">Prediksi Berhasil!</div>',
                unsafe_allow_html=True,
            )

            left, right = st.columns(2)
            left.metric(
                "Prediksi Identitas",
                f"{label} - {nama}" if nama else label
            )

            right.metric(
                "Tingkat Keyakinan",
                f"{confidence:.1%}"
            )

            st.subheader("Top 5 Prediksi")
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

            probability_df = (
                probability_df
                .sort_values("Probabilitas", ascending=False)
                .head(5)
            )

            class_labels = probability_df["Kelas"].tolist()

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
                        alt.Tooltip("Kelas:N"),
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
            Klasifikasi Tulisan Tangan berbasis Vision Transformer DINOv2
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