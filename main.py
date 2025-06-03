"""
Bu uygulama, farklı veri işleme ve şifreleme yaklaşımlarının performanslarını 
karşılaştırmak amacıyla oluşturulmuştur. Veri sıkıştırma (Huffman), boyut indirgeme (Autoencoder) 
ve şifreleme (AES, RSA) yöntemlerinin farklı kombinasyonlarının zaman, bellek kullanımı ve 
veri boyutu açısından performanslarını karşılaştırmalı olarak analiz etmektir.
"""


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
import heapq, psutil, os, time
from collections import defaultdict
import tracemalloc


# =========================
# HUFFMAN KODLAMA
# =========================
# Kayıpsız veri sıkıştırma yöntemidir. Sık tekrarlanan karakterlere daha kısa, 
# nadir olanlara daha uzun bit dizileri atar. Bu sayede veri boyutu azaltılır.
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char  # Karakter (veya byte değeri)
        self.freq = freq  # Frekans (kaç kez geçtiği)
        self.left = None 
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_map):  # Frekans haritasından Huffman ağacı oluşturma
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

# Huffman ağacından kod tablosunu çıkarma (karakter -> bit dizisi)
def build_codes(node, prefix="", code_map=None):
    if code_map is None:
        code_map = {}
    if node:
        if node.char is not None:
            code_map[node.char] = prefix
        build_codes(node.left, prefix + "0", code_map)
        build_codes(node.right, prefix + "1", code_map)
    return code_map

# Huffman ile sıkıştırma (veri -> bit dizisine -> baytlara)
def huffman_encode(data_bytes):
    freq_map = defaultdict(int)
    for b in data_bytes:
        freq_map[b] += 1
    root = build_huffman_tree(freq_map)
    code_map = build_codes(root)
    encoded_bits = ''.join(code_map[b] for b in data_bytes)
    # 8 bitlik bloklara bölebilmek için sıfırla doldurma (padding)
    padded_encoded_bits = encoded_bits + '0' * ((8 - len(encoded_bits) % 8) % 8)
    byte_array = bytearray()
    for i in range(0, len(padded_encoded_bits), 8):
        byte_array.append(int(padded_encoded_bits[i:i+8], 2))
    return bytes(byte_array)

# =========================
# AES / RSA
# =========================
# AES için veri bloklarını 16 byte'a pad etme
def pad(data_bytes):
    pad_len = 16 - (len(data_bytes) % 16)
    return data_bytes + bytes([pad_len]) * pad_len
# AES şifreleme (ECB modu). Blok şifreleme algoritmasıdır. Simetrik anahtarlı çalışır, 
# yani şifreleme ve çözme işlemleri aynı anahtarla yapılır. Hızlıdır ve özellikle büyük veri bloklarında etkilidir.
def encrypt_aes(data_bytes, key_bytes):
    cipher = AES.new(key_bytes, AES.MODE_ECB)
    return cipher.encrypt(pad(data_bytes))
# RSA ile veri şifreleme (uzunluk sınırlaması var!). Asimetrik bir şifreleme algoritmasıdır. 
# Açık ve özel anahtar çiftiyle çalışır. Daha güvenlidir fakat büyük verilerde yavaştır ve daha fazla kaynak kullanır.
def encrypt_rsa(data_bytes, rsa_key):
    cipher = PKCS1_OAEP.new(rsa_key)
    # RSA OAEP ile şifrelenebilecek maksimum veri miktarı sınırlı
    return cipher.encrypt(data_bytes[:rsa_key.size_in_bytes() - 42])

# =========================
# AUTOENCODER 
# =========================
# 540 boyutlu girdiyi 16 boyuta kadar sıkıştırıp geri çıkaran bir autoencoder. 
# Bir yapay sinir ağı modelidir. Girdi verisini daha küçük boyutlu bir temsile (encoding) dönüştürür 
# ve tekrar orijinal veriyi üretmeyi öğrenir. Bu projede boyut indirgeme ve veri sıkıştırma için kullanılmıştır.
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(540, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 540), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# =========================
# COMPARISON TESTS
# =========================
def run_test(use_autoencoder=True):
    # 540 özellikli rastgele veri üret
    X, _ = make_regression(n_samples=1000, n_features=540, noise=0.1, random_state=42)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

     # Autoencoder ile veri sıkıştırması yapılacaksa eğitim yap
    if use_autoencoder:
        autoencoder = Autoencoder()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
        dataloader = DataLoader(TensorDataset(X_tensor), batch_size=64, shuffle=True)
        for _ in range(20): # EPOCH
            for batch in dataloader:
                x = batch[0]
                output = autoencoder(x)
                loss = criterion(output, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        autoencoder.eval()
        with torch.no_grad():
            encoded_data = autoencoder.encoder(X_tensor).cpu().numpy()
    else:
        encoded_data = X_scaled.astype(np.float32)

    aes_key = get_random_bytes(16)
    rsa_key = RSA.generate(2048)
    results = []

    # İlk 50 veri noktası için test
    for i in range(50):
        vector = encoded_data[i].astype(np.float32).tobytes()
        vector_huff = huffman_encode(vector)

        for method in ["AES", "RSA"]:
            for compress in [False, True]:
                data = vector_huff if compress else vector
                #process = psutil.Process(os.getpid())
                #mem_before = process.memory_info().rss #/ 1024
                #start = time.perf_counter()
                
                
                tracemalloc.start()
                start = time.perf_counter()
                if method == "AES":
                    _ = encrypt_aes(data, aes_key)
                else:
                    _ = encrypt_rsa(data, rsa_key.publickey())
                    
                    
                end = time.perf_counter()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                    
                #end = time.perf_counter()
                #mem_after = process.memory_info().rss #/ 1024
                results.append({
                    "Autoencoder": "Var" if use_autoencoder else "Yok",
                    "Şifreleme": method,
                    "Sıkıştırma": "Huffman" if compress else "Yok",
                    "Zaman (s)": end - start,
                    "Bellek (Bytes)": peak - current,
                    "Veri boyutu": len(data),
                    "Anahtar Uzayı": 2 ** (aes_key.__len__() * 8) if method == "AES" else 2 ** rsa_key.size_in_bits()
                })

    
    return pd.DataFrame(results), pd.DataFrame(X_scaled)

# =========================
# STREAMLIT
# =========================
st.title("🔐 Şifreleme Karşılaştırma Uygulaması")

# Autoencoder kullanılarak ve kullanılmadan ayrı ayrı test yapılır
df1, original1 = run_test(use_autoencoder=True)
df2, original2 = run_test(use_autoencoder=False)

df = pd.concat([df1, df2])
original_data = pd.concat([original1, original2])

st.subheader("📌 Örnek Girdi Verisi")
st.write("Aşağıda modelin işlediği 540 boyutlu verilerden örnekler gösterilmektedir:")
st.dataframe(original_data.sample(5))

# Sonuçların gruplandırılmış ortalamalarını çıkar
st.subheader("📊 Ortalama Sonuçlar")
mean_df = df.groupby(["Autoencoder", "Şifreleme", "Sıkıştırma"]).agg({
    "Zaman (s)": "mean",
    "Bellek (Bytes)": "mean",
    "Veri boyutu": "mean",
    "Anahtar Uzayı": "first"   # Değişmediği için tek bir değer alınabilir
}).reset_index()

# Anahtar Uzayı çok büyük int olduğundan string'e dönüştürÜR
mean_df["Anahtar Uzayı"] = mean_df["Anahtar Uzayı"].astype(str)
st.dataframe(mean_df)

# Zaman ve bellek kullanımı grafikleri
st.subheader("🖼️ Zaman ve Bellek Kullanımı")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=mean_df, x="Şifreleme", y="Zaman (s)", hue="Autoencoder", ax=ax[0])
ax[0].set_title("Zaman Karşılaştırması")
sns.barplot(data=mean_df, x="Şifreleme", y="Bellek (Bytes)", hue="Autoencoder", ax=ax[1])
ax[1].set_title("Bellek Kullanımı")
st.pyplot(fig)