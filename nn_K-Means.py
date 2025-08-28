import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ----- 1. 特征提取函数 -----

# 提取时域特征
def extract_time_features(signal):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    kurt_val = kurtosis(signal)
    skew_val = skew(signal)
    peak_to_peak = np.ptp(signal)
    rms_val = np.sqrt(np.mean(signal**2))
    return [mean_val, std_val, kurt_val, skew_val, peak_to_peak, rms_val]

# 提取频域特征
def extract_freq_features(signal, fs):
    N = len(signal)
    freqs = rfftfreq(N, d=1/fs)
    fft_vals = np.abs(rfft(signal))
    fft_power = fft_vals**2

    # 主频
    main_freq = freqs[np.argmax(fft_power)]
    # 频谱质心
    spectral_centroid = np.sum(freqs * fft_power) / np.sum(fft_power)
    # 带宽
    bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_power) / np.sum(fft_power))
    # 频谱熵
    psd_norm = fft_power / np.sum(fft_power)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return [main_freq, spectral_centroid, bandwidth, spectral_entropy]

# 提取dB值特征
def extract_db_features(signal):
    power = np.mean(signal**2)
    db_val = 10 * np.log10(power + 1e-12)
    return [db_val]

# ----- 2. K-Means 聚类函数 -----

# K-Means 聚类
def kmeans_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans.labels_, kmeans

# ----- 3. 数据处理 -----

# 假设有多个通道数据
# all_channels: list of N numpy arrays representing signals from different channels
# fs: sampling rate
fs = 1000  # 示例采样率

# 假设的信号数据（示例随机数据，你可以替换为实际的信号）
N = 10  # 10个通道
T = 1000  # 每个通道的数据长度
np.random.seed(42)
all_channels = [np.random.randn(T) + np.sin(2 * np.pi * 50 * np.linspace(0, T/fs, T)) for _ in range(N)]

# 提取所有通道的时域、频域和dB值特征
features = []
for ch in all_channels:
    time_f = extract_time_features(ch)
    freq_f = extract_freq_features(ch, fs)
    db_f = extract_db_features(ch)
    features.append(time_f + freq_f + db_f)

# 特征标准化
X = StandardScaler().fit_transform(features)

# ----- 4. K-Means 聚类分析 -----

# 选择聚类簇数为3
labels_kmeans, kmeans_model = kmeans_clustering(X, n_clusters=3)

# ----- 5. 可视化聚类结果 -----

# 使用PCA降维到2D，方便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-Means聚类结果可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='rainbow')
plt.title('K-Means Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# ----- 6. 聚类结果输出 -----
print("K-Means Clustering Labels:", labels_kmeans)

# 打印聚类的中心点
print("Cluster Centers (in original feature space):")
print(kmeans_model.cluster_centers_)
