# main.py
# Çalıştırmak için: `streamlit run main.py`

import os
import io
import time
import requests
import heapq
import psutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from PIL import Image
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression

# ==============================================================================
# VQGAN / VQ-VAE için “taming-transformers” aracını import ediyoruz
# ==============================================================================
# Bu paket; önceden eğitilmiş VQGAN/VQ-VAE modellerini indirmenizi sağlar.
# Kurmak için terminale:
#   pip install torch torchvision taming-transformers-rom1504 omegaconf
#
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from torchvision import transforms

# ==============================================================================
# 1. COCO GÖRSELİ İNDİRME / YÜKLEME
# ==============================================================================

COCO_SAMPLE_URLS = [
    "https://images.cocodataset.org/val2017/000000000139.jpg",
    "https://images.cocodataset.org/val2017/000000000285.jpg",
    "https://images.cocodataset.org/val2017/000000000632.jpg",
]

def download_file(url, local_path):
    """Verilen URL'den dosyayı indirir ve local_path'e kaydeder."""
    if os.path.isfile(local_path):
        return local_path
        
    dir_path = os.path.dirname(local_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    response = requests.get(url, stream=True, verify = False)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path


def download_coco_images(url_list, save_dir="coco_samples"):
    os.makedirs(save_dir, exist_ok=True)
    local_paths = []
    for url in url_list:
        filename = os.path.join(save_dir, os.path.basename(url))
        download_file(url, filename)
        local_paths.append(filename)
    return local_paths

@st.cache_data(show_spinner=False)
def load_images_from_coco():
    """
    COCO_SAMPLE_URLS listesindeki görselleri indirir ve PIL objesi listesi döner.
    """
    paths = download_coco_images(COCO_SAMPLE_URLS)
    images = [Image.open(p).convert("RGB") for p in paths]
    return images, paths

# ==============================================================================
# 2. HUFFMAN KODLAMA
# ==============================================================================

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_map):
    heap = [HuffmanNode(char, freq) for char, freq in freq_map.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]

def build_codes(node, prefix="", code_map=None):
    if code_map is None:
        code_map = {}
    if node:
        if node.char is not None:
            code_map[node.char] = prefix
        build_codes(node.left, prefix + "0", code_map)
        build_codes(node.right, prefix + "1", code_map)
    return code_map

def huffman_encode(data_bytes):
    freq_map = defaultdict(int)
    for b in data_bytes:
        freq_map[b] += 1
    root = build_huffman_tree(freq_map)
    code_map = build_codes(root)
    encoded_bits = ''.join(code_map[b] for b in data_bytes)
    padded = encoded_bits + '0' * ((8 - len(encoded_bits) % 8) % 8)
    byte_array = bytearray()
    for i in range(0, len(padded), 8):
        byte_array.append(int(padded[i:i+8], 2))
    return bytes(byte_array), root, len(encoded_bits)  # ağacı ve orijinal bit uzunluğunu döndür

def huffman_decode(encoded_bytes, root, bit_length):
    bits = ''.join(f'{byte:08b}' for byte in encoded_bytes)
    bits = bits[:bit_length]  # padding'i çıkar
    decoded = bytearray()
    node = root
    for bit in bits:
        node = node.left if bit == '0' else node.right
        if node.char is not None:
            decoded.append(node.char)
            node = root
    return bytes(decoded)


# ==============================================================================
# 3. ŞİFRELEME (AES ve RSA)
# ==============================================================================

def pad_bytes(data_bytes):
    pad_len = 16 - (len(data_bytes) % 16)
    return data_bytes + bytes([pad_len]) * pad_len

def encrypt_aes(data_bytes, key_bytes):
    cipher = AES.new(key_bytes, AES.MODE_ECB)
    return cipher.encrypt(pad_bytes(data_bytes))

def encrypt_rsa(data_bytes, rsa_pubkey):
    cipher = PKCS1_OAEP.new(rsa_pubkey)
    max_len = rsa_pubkey.key_size // 8 - 42
    return cipher.encrypt(data_bytes[:max_len])

# ==============================================================================
# 4. VQGAN / VQ-VAE YÜKLEME (Elle YAML + Checkpoint)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_vqvae_model():
    config_url = "https://drive.google.com/uc?export=download&id=1ikuE7ZfsCdgYN6fSDEDVxBRglMUmy7SS"
    ckpt_url   = "https://drive.google.com/uc?export=download&id=1kw4VA8h6P66YJ2uo7YiGEcYQNJ_R3Rbr"
    config_local = download_file(config_url, "model.yaml")
    ckpt_local   = download_file(ckpt_url,   "last.ckpt")

    config = OmegaConf.load(config_local)
    model = VQModel(**config.model.params)
    ckpt_dict = torch.load(ckpt_local, map_location="cpu", weights_only=False)
    state_dict = ckpt_dict.get("state_dict", ckpt_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ==============================================================================
# 5. YARDIMCI FONKSİYONLAR
# ==============================================================================

def pil_to_jpeg_bytes(img, quality=75):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    data = buf.getvalue()
    buf.close()
    return data

def img_to_array(img, resize_to=None):
    """
    PIL -> numpy array [0,1]
    Eğer resize_to=(128,128) verilirse yeniden boyutlandırır.
    """
    if resize_to is not None:
        img = img.resize(resize_to, Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def array_to_pil(arr):
    """
    [0,1] aralığında numpy array -> PIL.Image
    """
    arr_uint8 = (arr * 255).clip(0,255).astype(np.uint8)
    return Image.fromarray(arr_uint8)

def measure_memory(func, *args, **kwargs):
    """
    Bir fonksiyon çağrısı öncesi ve sonrası bellek farkını (MB) ve süreyi döner.
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start_time
    mem_after = process.memory_info().rss
    return result, elapsed, (mem_after - mem_before) / 1024**2

# ==============================================================================
# 6. SIKIŞTIRMA ve METRİKLERİ HESAPLAMA
# ==============================================================================

def evaluate_compression(num_samples=5):
    images, paths = load_images_from_coco()
    vqvae = load_vqvae_model()
    records = []

    for img, path in zip(images, paths):
        # Orijinal görüntüyü PNG formatında hafızada tutup bayt sayısını al
        with io.BytesIO() as buf:
            img.save(buf, format="PNG")
            orig_bytes = buf.getvalue()
        orig_size = len(orig_bytes)

        # Görseli [0,1] aralıklı numpy dizisine çevir
        arr_full = img_to_array(img)

        # ----------- JPEG SIKIŞTIRMA -----------
        jpeg_bytes, jpeg_time, jpeg_mem = measure_memory(pil_to_jpeg_bytes, img, 75)
        jpeg_size = len(jpeg_bytes)
        jpeg_img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        jpeg_arr = img_to_array(jpeg_img)

        # ----------- HUFFMAN SIKIŞTIRMA -----------     
        raw_array = (arr_full * 255).clip(0, 255).astype(np.uint8)
        raw_bytes = raw_array.tobytes()
        
        huff_encoded, huff_tree, bit_length = huffman_encode(raw_bytes)
        huff_size = len(huff_encoded)
        
        _, huff_time, huff_mem = measure_memory(huffman_encode, raw_bytes)
        _, huff_dec_time, huff_dec_mem = measure_memory(huffman_decode, huff_encoded, huff_tree, bit_length)
        
        huff_decoded_bytes = huffman_decode(huff_encoded, huff_tree, bit_length)
        huff_decoded_array = np.frombuffer(huff_decoded_bytes, dtype=np.uint8).reshape(arr_full.shape)
        
        # normalize again to [0,1] for comparison
        huff_decoded_array = huff_decoded_array.astype(np.float32) / 255.0
        
        psnr_huff = psnr(arr_full, huff_decoded_array, data_range=1.0)
        if np.isnan(psnr_huff):
            psnr_huff = float("inf")
        
        ssim_huff = ssim(arr_full, huff_decoded_array, channel_axis=2, data_range=1.0)


        # ----------- VQ-VAE SIKIŞTIRMA -----------
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        input_tensor = transform(img).unsqueeze(0)  # (1,3,128,128)

        # Encode (latent kod ve codebook indeksleri çıkar)
        with torch.no_grad():
            #z, z_indices = vqvae.encode(input_tensor)
            z_q, z_indices, z_e = vqvae.encode(input_tensor)
                        

        vq_bytes = z_indices.cpu().numpy().astype(np.int32).tobytes()
        vq_size = len(vq_bytes)

        # Encode süresi ve bellek ölçümü
        _, vq_enc_time, vq_enc_mem = measure_memory(vqvae.encode, input_tensor)
        # Decode süresi ve bellek ölçümü
        _, vq_dec_time, vq_dec_mem = measure_memory(vqvae.decode, z_q)

        # Decode sonrası yeniden oluşturulan görüntü
        with torch.no_grad():
            recon = vqvae.decode(z_q).squeeze(0).permute(1, 2, 0).cpu().numpy()
        # (128,128,3) aralıklı [0,1]

        # Orijinal görüntüyü 128x128'e küçült ve karşılaştırma için hazırlık
        arr_resized = img.resize((128, 128))
        arr_resized = np.array(arr_resized).astype(np.float32) / 255.0

        # PSNR / SSIM hesaplamaları
        psnr_jpeg = psnr(arr_full, jpeg_arr, data_range=1.0)
        ssim_jpeg = ssim(arr_full, jpeg_arr, channel_axis=2, data_range=1.0)

        psnr_vq = psnr(arr_resized, recon, data_range=1.0)
        ssim_vq = ssim(arr_resized, recon, channel_axis=2, data_range=1.0)

        basename = os.path.basename(path)

        records.append({
            "Image": basename,
            "Method": "JPEG",
            "CompressionRatio": orig_size / jpeg_size,
            "TimeEncode": jpeg_time,
            "TimeDecode": 0.0,
            "MemoryEncode_MB": jpeg_mem,
            "MemoryDecode_MB": 0.0,
            "PSNR": psnr_jpeg,
            "SSIM": ssim_jpeg
        })
        records.append({
            "Image": basename,
            "Method": "Huffman",
            "CompressionRatio": orig_size / huff_size,
            "TimeEncode": huff_time,
            "TimeDecode": huff_dec_time,
            "MemoryEncode_MB": huff_mem,
            "MemoryDecode_MB": huff_dec_mem,
            "PSNR": psnr_huff,
            "SSIM": ssim_huff
        })

        records.append({
            "Image": basename,
            "Method": "VQ-VAE",
            "CompressionRatio": orig_size / vq_size,
            "TimeEncode": vq_enc_time,
            "TimeDecode": vq_dec_time,
            "MemoryEncode_MB": vq_enc_mem,
            "MemoryDecode_MB": vq_dec_mem,
            "PSNR": psnr_vq,
            "SSIM": ssim_vq
        })

    return pd.DataFrame(records)

# ==============================================================================
# 7. STREAMLIT ARAYÜZÜ
# ==============================================================================

st.title("📊 COCO Görselleri Üzerinde Sıkıştırma Karşılaştırması")

st.write(
    "Bu uygulama, COCO’dan çektiğimiz örnek görseller üzerinde "
    "JPEG, Huffman ve VQ-VAE yöntemlerini karşılaştırır. "
    "Sıkıştırma oranı, PSNR, SSIM, kodlama/çözme süreleri ve bellek kullanımını görselleştirir."
)

@st.cache_data(show_spinner=True)
def get_metrics():
    return evaluate_compression(num_samples=5)

df_metrics = get_metrics()

st.subheader("🔖 Tüm Metrik Sonuçları")
st.dataframe(df_metrics)

mean_df = df_metrics.groupby("Method").mean(numeric_only=True).reset_index()

st.subheader("📈 Ortalama Sonuçlar (Her Bir Yöntem İçin)")
st.table(mean_df)

st.subheader("🎨 Sıkıştırma Oranı Karşılaştırması")
fig1, ax1 = plt.subplots()
sns.barplot(data=mean_df, x="Method", y="CompressionRatio", ax=ax1)
ax1.set_ylabel("Sıkıştırma Oranı")
st.pyplot(fig1)

st.subheader("🎨 PSNR Karşılaştırması")
fig2, ax2 = plt.subplots()
sns.barplot(data=mean_df, x="Method", y="PSNR", ax=ax2)
ax2.set_ylabel("PSNR (dB)")
st.pyplot(fig2)

st.subheader("🎨 SSIM Karşılaştırması")
fig3, ax3 = plt.subplots()
sns.barplot(data=mean_df, x="Method", y="SSIM", ax=ax3)
ax3.set_ylabel("SSIM")
st.pyplot(fig3)

st.subheader("🎨 Kodlama Süreleri Karşılaştırması")
fig4, ax4 = plt.subplots()
sns.barplot(data=mean_df, x="Method", y="TimeEncode", ax=ax4)
ax4.set_ylabel("Kodlama Süresi (s)")
st.pyplot(fig4)

st.subheader("🎨 Bellek Kullanımı (Encode) Karşılaştırması")
fig5, ax5 = plt.subplots()
sns.barplot(data=mean_df, x="Method", y="MemoryEncode_MB", ax=ax5)
ax5.set_ylabel("Bellek Kullanımı (MB)")
st.pyplot(fig5)
