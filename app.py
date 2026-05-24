import streamlit as st
import tensorflow as tf
from PIL import Image
import os
import numpy as np


# Import seluruh mesin utama dari utils.py
from utils import (
    preprocess_image, 
    predict_image, 
    make_gradcam_heatmap, 
    generate_gradcam_overlay,
    build_hybrid_model,
    build_efficientnet_model,
    build_vit_model,
    CLASS_NAMES,
)

# ==========================================
# 1. PENGATURAN HALAMAN WEB
# ==========================================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Pisang",
    page_icon="🍌",
    layout="wide"
)

# ==========================================
# 2. FUNGSI PEMUATAN MODEL (ADAPTASI .WEIGHTS.H5)
# ==========================================
@st.cache_resource
def load_disease_model(model_choice):
    """
    Membangun arsitektur (rumah) dan memuat bobot (perabotan)
    berdasarkan pilihan pengguna di sidebar.
    """
    if model_choice == "Skenario A (EfficientNetV2B0)":
        # 1. Bangun arsitektur Skenario A
        model = build_efficientnet_model()
        # 2. Masukkan bobot Skenario A
        weights_path = os.path.join("models", "Bobot_Skenario_A_EfficientNetV2.weights.h5")
        model.load_weights(weights_path)
        return model
    
    elif model_choice == "Skenario B (ViT Tunggal)": 
        model = build_vit_model()
        weights_path = os.path.join("models", "Bobot_Skenario_B_ViT.weights.h5") 
        model.load_weights(weights_path)
        return model
        
    elif model_choice == "Skenario C (Hibrida EffNet-ViT)":
        # 1. Bangun arsitektur Skenario C
        model = build_hybrid_model()
        # 2. Masukkan bobot Skenario C
        weights_path = os.path.join("models", "Bobot_Skenario_C_Hibrida_EffNet_ViT.weights.h5") 
        model.load_weights(weights_path)
        return model
        
    return None

# ==========================================
# 3. ANTARMUKA (UI) UTAMA
# ==========================================
st.title(" Sistem Deteksi Penyakit Daun Pisang")
st.markdown("**Analisis Komparatif: *Deep Learning* vs Arsitektur Hibrida (*Explainable AI*)**")
st.markdown("---")

# SIDEBAR PENGATURAN
st.sidebar.header("⚙️ Pengaturan Analisis")
model_option = st.sidebar.selectbox(
    "Pilih Arsitektur Model:",
    [
        "Skenario A (EfficientNetV2B0)", 
        "Skenario B (ViT Tunggal)",
        "Skenario C (Hibrida EffNet-ViT)"
    ]
)

# LOGIKA PEMBATASAN GRAD-CAM
# ViT Tunggal tidak memiliki layer konvolusi, jadi Grad-CAM harus dimatikan.
is_gradcam_supported = True
if "ViT Tunggal" in model_option:
    is_gradcam_supported = False

# COBA MEMUAT MODEL YANG DIPILIH
try:
    with st.spinner(f"Memuat {model_option}..."):
        aktif_model = load_disease_model(model_option)
    st.sidebar.success(f" {model_option.split(' ')[0]} berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"Gagal memuat model. Pastikan file bobot ada di folder 'models'. Error: {e}")
    st.stop() # Hentikan aplikasi jika file bobot tidak ditemukan

# AREA UPLOAD GAMBAR
st.subheader("Unggah Foto Daun Pisang")
uploaded_file = st.file_uploader("Pilih file gambar (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bagi layar menjadi dua kolom
    col1, col2 = st.columns(2)
    
    # Prapemrosesan Gambar
    img_pil, img_array, img_tensor = preprocess_image(uploaded_file)
    
    # Bagian Kiri: Tampilkan Input & Tombol
    with col1:
        st.markdown("### Citra Input")
        st.image(img_pil, use_column_width=True)
        tombol_analisis = st.button("🔍 Mulai Analisis Penyakit", use_container_width=True, type="primary")

    # Jika tombol ditekan, jalankan proses ini
    if tombol_analisis:
        
        # PROSES 1: Tampilkan Teks di Kolom Kiri (di bawah tombol)
        with col1:
            with st.spinner("Model sedang menganalisis..."):
                class_name, confidence, preds, pred_index = predict_image(aktif_model, img_tensor)
                
                st.markdown("###  Hasil Klasifikasi")
                if "Healthy" in class_name:
                    st.success(f"🌿 **Status: TANAMAN SEHAT**\n\n**Prediksi Utama:** {class_name} ({confidence:.2f}%)")
                else:
                    st.error(f"⚠️ **Status: TERDETEKSI PENYAKIT!**\n\n**Prediksi Utama:** {class_name} ({confidence:.2f}%)")
                    
                # Kemungkinan lainnya untuk SEMUA kasus (sakit/sehat)
                st.markdown("#### Kemungkinan Lainnya:")
                probabilitas = preds[0]
                urutan_index = np.argsort(probabilitas)[::-1]
                
                for i in range(1, 3): # Ambil ranking 2 dan 3
                    idx = urutan_index[i]
                    nama_kelas_lain = CLASS_NAMES[idx]
                    nilai_prob = probabilitas[idx] * 100
                    # Tampilkan tanpa syarat > 1%
                    st.write(f"- {nama_kelas_lain}: {nilai_prob:.4f}%")
                            
                st.info("💡 **Tips untuk Petani:** Peta Grad-CAM di sebelah kanan menunjukkan area bercak yang dilihat oleh kecerdasan buatan. Gunakan ini sebagai referensi visual Anda. " \
                "Namun, jika bentuk bercak tidak sesuai dengan gejala khas dari tebakan utama, harap pertimbangkan kemungkinan penyakit lainnya di atas.")

        # PROSES 2: Tampilkan Gambar Grad-CAM di Kolom Kanan
        with col2:
            st.markdown("### Analisis Area Fokus")
            
            if is_gradcam_supported:
                with st.spinner("Mengekstrak peta gradien visual (Grad-CAM)..."):
                    try:
                        last_conv_layer_name = 'top_activation' 
                        heatmap = make_gradcam_heatmap(img_tensor, aktif_model, last_conv_layer_name, pred_index)
                        cam_image = generate_gradcam_overlay(img_array, heatmap)
                        
                        st.image(cam_image, caption=f"Peta Panas Grad-CAM untuk memprediksi {class_name}", use_column_width=True)
                    except Exception as ve:
                        st.warning(f"Tidak dapat menghasilkan Grad-CAM. Error: {ve}")
            else:
                # Jika ViT yang dipilih, Grad-CAM tidak dijalankan.
                st.warning("⚠️ **Catatan Sistem:** Skenario B (ViT Tunggal) tidak mendukung pemetaan Grad-CAM karena arsitekturnya murni menggunakan mekanisme 'Self-Attention' tanpa lapisan konvolusi spasial.")
                st.image(img_pil, caption="Citra Input Asli (Tanpa Pemetaan Atensi)", use_column_width=True)

                

else:
    st.info("Silakan unggah foto daun pisang untuk memulai proses deteksi.")

st.markdown("---")
st.caption("Aplikasi dikembangkan untuk keperluan demonstrasi Sidang Skripsi.")