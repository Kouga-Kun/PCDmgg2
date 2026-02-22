import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def quantize_uniform(img, levels):
    """Kuantisasi Uniform: Membagi 256 level menjadi n level."""
    shift = 256 // levels
    quantized = (img // shift) * shift
    return quantized

def quantize_non_uniform(img, k=4):
    """Kuantisasi Non-Uniform: Menggunakan K-Means Clustering."""
    data = img.reshape((-1, 1 if len(img.shape)==2 else 3))
    data = np.float32(data)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))

def process_analysis(image_path, label):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return None
    
    # 1. Konversi Ruang Warna & Waktu Komputasi
    start_time = time.time()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    ctime = (time.time() - start_time) * 1000 # ms

    # 2. Kuantisasi
    q_uni = quantize_uniform(gray, 8) # 8 level (3-bit)
    q_non = quantize_non_uniform(gray, 8)

    # 3. Hitung Parameter Teknis
    mem_orig = img_bgr.nbytes / 1024 # KB
    mem_quant = q_uni.nbytes / 1024
    ratio = mem_orig / mem_quant

    return {
        'orig': cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        'gray': gray, 'hsv': hsv, 'lab': lab,
        'q_uni': q_uni, 'q_non': q_non,
        'ctime': ctime, 'ratio': ratio, 'label': label
    }

# Simulasi load 3 kondisi cahaya
conditions = ['terang.jpeg', 'normal.jpeg', 'redup.jpeg']
all_results = []

for cond in conditions:
    # Jika file tidak ada, lewati
    res = process_analysis(cond, cond.split('.')[0])
    if res: all_results.append(res)

# --- VISUALISASI UNTUK LAPORAN ---
if all_results:
    fig, axes = plt.subplots(len(all_results), 4, figsize=(15, 10))
    for i, data in enumerate(all_results):
        axes[i, 0].imshow(data['orig']); axes[i,0].set_title(f"Original ({data['label']})")
        axes[i, 1].imshow(data['gray'], cmap='gray'); axes[i,1].set_title("Grayscale")
        axes[i, 2].imshow(data['q_uni'], cmap='gray'); axes[i,2].set_title("Uniform Quant (8)")
        axes[i, 3].hist(data['q_uni'].ravel(), bins=16); axes[i,3].set_title("Histogram Q-Uni")
    
    plt.tight_layout()
    plt.show()

    # Print Parameter Teknis untuk Tabel Laporan
    print(f"{'Kondisi':<10} | {'Waktu Konversi':<15} | {'Rasio Kompresi':<15}")
    for d in all_results:
        print(f"{d['label']:<10} | {d['ctime']:.4f} ms     | {d['ratio']:.2f}x")