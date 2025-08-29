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
    # 计算信号的均值
    mean_val = np.mean(signal)
    # 计算信号的标准差
    std_val = np.std(signal)
    # 计算信号的峭度（kurtosis），反映信号分布的尾部特征
    kurt_val = kurtosis(signal)
    # 计算信号的偏度（skewness），反映信号分布的对称性
    skew_val = skew(signal)
    # 计算信号的峰峰值（即最大值与最小值的差）
    peak_to_peak = np.ptp(signal)
    # 计算信号的均方根值（Root Mean Square）
    rms_val = np.sqrt(np.mean(signal**2))
    # 返回一个包含所有时域特征的列表
    return [mean_val, std_val, kurt_val, skew_val, peak_to_peak, rms_val]

# 提取频域特征
def extract_freq_features(signal, fs):
    N = len(signal)  # 获取信号长度
    # 计算信号的频率范围
    freqs = rfftfreq(N, d=1/fs)
    # 计算信号的FFT（快速傅里叶变换），返回信号的频谱
    fft_vals = np.abs(rfft(signal))
    # 计算频谱的功率
    fft_power = fft_vals**2

    # 计算信号的主频（最大功率对应的频率）
    main_freq = freqs[np.argmax(fft_power)]
    # 计算频谱质心，表示频谱的“重心”
    spectral_centroid = np.sum(freqs * fft_power) / np.sum(fft_power)
    # 计算信号的带宽，反映频谱的宽度
    bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_power) / np.sum(fft_power))
    # 计算频谱的熵（谱熵），反映信号的复杂度
    psd_norm = fft_power / np.sum(fft_power)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    # 返回频域特征
    return [main_freq, spectral_centroid, bandwidth, spectral_entropy]

# 提取dB值特征
def extract_db_features(signal):
    # 计算信号的功率（均方值）
    power = np.mean(signal**2)
    # 计算信号的dB值（分贝）
    db_val = 10 * np.log10(power + 1e-12)
    # 返回dB特征
    return [db_val]

# ----- 2. K-Means 聚类函数 -----

# K-Means 聚类
def kmeans_clustering(X, n_clusters=3): # 使用 KMeans 算法对数据进行聚类，指定簇数为 n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)# 返回聚类标签和 KMeans 模型
    return kmeans.labels_, kmeans

# ----- 3. 数据处理 -----

# 假设有多个通道数据
# all_channels: 代表多个通道信号的数据，每个信号为一个 NumPy 数组
# fs: 信号的采样率
fs = 1000  # 示例采样率

# 假设的信号数据（生成示例信号：高斯噪声 + 50Hz 正弦波）
N = 10  # 10个通道
T = 1000  # 每个通道的数据长度
np.random.seed(42)  # 设置随机种子，确保可复现性
# 生成10个通道信号，信号由高斯噪声与50Hz正弦波叠加
all_channels = [np.random.randn(T) + np.sin(2 * np.pi * 50 * np.linspace(0, T/fs, T)) for _ in range(N)]

# 提取所有通道的时域、频域和dB值特征
features = []
for ch in all_channels: # 提取每个通道的时域特征
    time_f = extract_time_features(ch) # 提取每个通道的频域特征
    freq_f = extract_freq_features(ch, fs) # 提取每个通道的dB值特征
    db_f = extract_db_features(ch)
    features.append(time_f + freq_f + db_f)

# 特征标准化
X = StandardScaler().fit_transform(features)  # 对特征进行标准化处理

# ----- 4. K-Means 聚类分析 -----

# 设置聚类簇数为3
labels_kmeans, kmeans_model = kmeans_clustering(X, n_clusters=3)
# ----- 5. 可视化聚类结果 -----
# 使用PCA降维到2D，方便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)  # 将特征降维为二维

# K-Means聚类结果可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='rainbow')
plt.title('K-Means Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# ----- 6. 聚类结果输出 -----
# 输出K-Means聚类标签
print("K-Means Clustering Labels:", labels_kmeans)
print("Cluster Centers (in original feature space):")
print(kmeans_model.cluster_centers_)
