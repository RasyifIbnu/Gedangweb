import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import keras_hub
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# ==========================================
# 1. KONSTANTA
# ==========================================
IMG_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ['Cordana', 'Healthy', 'Panama Disease', 'Yellow and Black Sigatoka'] 

# ==========================================
# 2. CUSTOM LAYER & ARSITEKTUR
# ==========================================
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(rate),
            layers.Dense(embed_dim),
            layers.Dropout(rate)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None, mask=None, **kwargs):
        if training is None:
            training = False

        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

def build_hybrid_model():
    """Merakit ulang arsitektur sesuai dengan yang ada di Google Colab"""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # 1. EfficientNetV2 Backbone
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights=None, input_tensor=inputs) # weights=None karena kita load bobot sendiri nanti
    x = backbone.output  

    # 2. Reshape → Patch Embedding
    _, h, w, c = x.shape
    x = layers.Reshape((h * w, c))(x) 

    embed_dim = 256
    x = layers.Dense(embed_dim)(x)    

    # 3. Positional Encoding
    positions = tf.range(start=0, limit=h * w, delta=1)
    pos_emb = layers.Embedding(input_dim=h * w, output_dim=embed_dim)(positions)
    x = x + pos_emb

    # 4. Transformer Encoder
    x = TransformerEncoder(
        embed_dim=embed_dim,
        num_heads=8,
        ff_dim=embed_dim * 4, 
        rate=0.3
    )(x)

    # 5. MLP Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax',
                           kernel_regularizer=regularizers.l2(0.01))(x)

    return models.Model(inputs, outputs, name="Skenario_C_Hibrida_EffNet_ViT")


# SKENARIO A: EFFICIENTNETV2 TUNGGAL
def build_efficientnet_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Mengambil backbone bawaan pabrik (Transfer Learning)
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inputs)
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)

    # Sabuk Pengaman Anti-Overfitting
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)
    return models.Model(inputs, outputs, name="Skenario_A_EfficientNetV2")


# SKENARIO B: VISION TRANSFORMER (ViT) TUNGGAL
def build_vit_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Mengambil backbone ViT bawaan pabrik
    vit_backbone = keras_hub.models.ViTBackbone.from_preset("vit_base_patch16_224_imagenet")
    x = vit_backbone(inputs)
    x = layers.GlobalAveragePooling1D()(x)

    # Sabuk Pengaman Anti-Overfitting
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)
    return models.Model(inputs, outputs, name="Skenario_B_ViT")



# ==========================================
# 3. FUNGSI PRAPEMROSESAN & INFERENSI
# ==========================================
def preprocess_image(image_file):
    """Menggunakan Keras agar 100% sama dengan Colab"""
    # Streamlit membaca file sebagai objek memori, load_img Keras bisa menerimanya
    img_pil = tf.keras.preprocessing.image.load_img(image_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img_pil)
    
    # Normalisasi [0, 1]
    img_array_norm = img_array / 255.0
    img_tensor = np.expand_dims(img_array_norm, axis=0)
    
    return img_pil, img_array, img_tensor

def predict_image(model, img_tensor):
    preds = model.predict(img_tensor)
    pred_index = np.argmax(preds[0])
    confidence = preds[0][pred_index] * 100
    class_name = CLASS_NAMES[pred_index]
    return class_name, confidence, preds, pred_index

# ==========================================
# 4. FUNGSI GRAD-CAM
# ==========================================
def make_gradcam_heatmap(img_tensor, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    heatmap /= max_val
    return heatmap.numpy()

def generate_gradcam_overlay(original_img_array, heatmap, alpha=0.8):
    " Heatmap menggunakan logika Matplotlib (Sama dengan Colab)"
    # Di Colab, menggabungkan heatmap dengan gambar yang sudah dinormalisasi (0-1)
    img_norm = original_img_array / 255.0
    
    img = np.uint8(255 * img_norm)
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Menggunakan Matplotlib Jet seperti di Colab
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    # Resize menggunakan Keras seperti di Colab
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    return superimposed_img