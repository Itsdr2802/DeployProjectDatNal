import streamlit as st
import numpy as np
import joblib

# ----------------------------------------------------
# 1. Load model & scaler
# ----------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Load model SVM dan StandardScaler yang sudah disimpan
    dari notebook (svm_model.pkl dan scaler.pkl).
    """
    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()

# ----------------------------------------------------
# 2. Mapping kategori -> angka (sesuai LabelEncoder)
# ----------------------------------------------------
sex_map = {"Female": 0, "Male": 1}

# LabelEncoder urut alfabet: ['ASY', 'ATA', 'NAP', 'TA']
cp_map = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}

# LabelEncoder urut alfabet: ['LVH', 'Normal', 'ST']
restecg_map = {"LVH": 0, "Normal": 1, "ST": 2}

# LabelEncoder urut alfabet: ['N', 'Y']
exang_map = {"No": 0, "Yes": 1}

# LabelEncoder urut alfabet: ['Down', 'Flat', 'Up']
slope_map = {"Down": 0, "Flat": 1, "Up": 2}

# FastingBS: 0/1 (<=120, >120)
fasting_map = {"≤ 120 mg/dL": 0, "> 120 mg/dL": 1}

# Target mapping (HeartDisease: 0 = No, 1 = Yes)
target_map = {
    0: "Tidak Mengalami Penyakit Jantung",
    1: "Mengalami Penyakit Jantung",
}


# ----------------------------------------------------
# 3. Fungsi untuk buat fitur & prediksi
# ----------------------------------------------------
def build_feature_vector(
    age,
    sex_label,
    chest_pain_label,
    resting_bp,
    chol,
    fasting_label,
    rest_ecg_label,
    max_hr,
    exang_label,
    oldpeak,
    st_slope_label,
):
    """
    Susun fitur dalam urutan yang sama seperti training:
    ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
     'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
     'ST_Slope']
    Lalu scale hanya kolom numerik:
    ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    """
    # 1. Mapping kategorikal -> angka
    sex = sex_map[sex_label]
    cp = cp_map[chest_pain_label]
    fasting = fasting_map[fasting_label]
    restecg = restecg_map[rest_ecg_label]
    exang = exang_map[exang_label]
    slope = slope_map[st_slope_label]

    # 2. Susun fitur mentah (belum discale)
    features = [
        float(age),        # Age (numeric)
        float(sex),        # Sex (encoded)
        float(cp),         # ChestPainType (encoded)
        float(resting_bp), # RestingBP (numeric)
        float(chol),       # Cholesterol (numeric)
        float(fasting),    # FastingBS (encoded)
        float(restecg),    # RestingECG (encoded)
        float(max_hr),     # MaxHR (numeric)
        float(exang),      # ExerciseAngina (encoded)
        float(oldpeak),    # Oldpeak (numeric)
        float(slope),      # ST_Slope (encoded)
    ]

    # 3. Scale hanya fitur numerik (urutan num_cols):
    # num_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    numeric_indices = [0, 3, 4, 7, 9]  # posisi Age, RestingBP, Cholesterol, MaxHR, Oldpeak

    numeric_values = [features[i] for i in numeric_indices]
    # scaler.fit sudah dilakukan di notebook, di sini hanya transform
    scaled_numeric = scaler.transform([numeric_values])[0]

    # Masukkan kembali nilai yang sudah di-scale
    for idx, val in zip(numeric_indices, scaled_numeric):
        features[idx] = val

    # 4. Kembalikan sebagai array (1, n_features)
    return np.array(features).reshape(1, -1)


def predict_heart_disease(features_array):
    """
    Prediksi HeartDisease (0/1) + probabilitas.
    """
    y_pred = model.predict(features_array)[0]
    # karena probability=True di SVC
    y_proba = model.predict_proba(features_array)[0]
    return int(y_pred), y_proba


# ----------------------------------------------------
# 4. UI Streamlit
# ----------------------------------------------------
def main():
    st.set_page_config(page_title="Heart Disease Prediction - SVM", page_icon="🫀")
    st.title("Heart Disease Risk Prediction 🫀")
    st.write(
        "Aplikasi ini menggunakan model **SVM**"
        "untuk memprediksi risiko penyakit jantung (HeartDisease)."
    )

    st.markdown("---")
    st.subheader("Masukkan Data Pasien")

    # Layout: 3 kolom
    col1, col2, col3 = st.columns(3)

    # ------------------ Input numerik ------------------
    with col1:
        age = st.number_input("Age (tahun)", min_value=20, max_value=100, value=50, step=1)
        resting_bp = st.number_input("Resting BP (mmHg)", min_value=0, max_value=250, value=130, step=1)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=800, value=200, step=1)

    with col2:
        max_hr = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=240, value=140, step=1)
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        fasting_label = st.radio(
            "Fasting Blood Sugar",
            options=["≤ 120 mg/dL", "> 120 mg/dL"],
            horizontal=False,
        )

    # ------------------ Input kategorikal ------------------
    with col3:
        sex_label = st.selectbox("Sex", options=["Female", "Male"])
        chest_pain_label = st.selectbox(
            "Chest Pain Type",
            options=["ASY", "ATA", "NAP", "TA"],
            help="ASY: Asymptomatic, ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina",
        )
        rest_ecg_label = st.selectbox("Resting ECG", options=["LVH", "Normal", "ST"])
        exang_label = st.selectbox("Exercise-induced Angina", options=["No", "Yes"])
        st_slope_label = st.selectbox("ST Slope", options=["Down", "Flat", "Up"])

    st.markdown("---")

    if st.button("Prediksi"):
        # 1. Bangun fitur
        X_input = build_feature_vector(
            age=age,
            sex_label=sex_label,
            chest_pain_label=chest_pain_label,
            resting_bp=resting_bp,
            chol=chol,
            fasting_label=fasting_label,
            rest_ecg_label=rest_ecg_label,
            max_hr=max_hr,
            exang_label=exang_label,
            oldpeak=oldpeak,
            st_slope_label=st_slope_label,
        )

        # 2. Prediksi
        y_pred, y_proba = predict_heart_disease(X_input)
        label_text = target_map.get(y_pred, str(y_pred))

        st.subheader("Hasil Prediksi")
        st.write(f"**Prediksi:** {label_text}")
        st.write(f"Label model (0 = tidak sakit, 1 = sakit): **{y_pred}**")

        # Probabilitas untuk masing-masing kelas
        st.subheader("Probabilitas Prediksi")
        st.write(f"Probabilitas Tidak Mengalami Penyakit Jantung (class 0): **{y_proba[0]:.3f}**")
        st.write(f"Probabilitas Mengalami Penyakit Jantung (class 1): **{y_proba[1]:.3f}**")

        # Interpretasi singkat
        st.markdown("---")
        st.markdown("### Interpretasi Singkat")
        if y_pred == 1:
            st.warning(
                "Model memprediksi pasien **BERISIKO mengalami penyakit jantung**. "
                "Ini bukan diagnosis medis, tapi bisa dijadikan bahan pertimbangan untuk pemeriksaan lebih lanjut."
            )
        else:
            st.success(
                "Model memprediksi pasien **tidak mengalami penyakit jantung**. "
                "Tetap jaga pola hidup sehat dan lakukan cek rutin."
            )


if __name__ == "__main__":
    main()
