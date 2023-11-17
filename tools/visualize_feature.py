import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


# 假设face_features是您的人脸特征数组，它的尺寸是(num_samples, num_features)
# 且labels是同样长度的标签数组，表示每个特征向量属于的ID。
# load features and labels
import pickle
feature_path = "logs/distill/face/base_12800_flat/results.pkl"
with open(feature_path, 'rb') as f:
    face_features_dict = pickle.load(f)
print(face_features_dict.keys())
# 载入您的特征和标签
face_features = face_features_dict["feature"] # 这里应该是一个numpy数组，形式可能是(num_samples, num_features)
labels = face_features_dict["label"] # 这里应该是一个numpy数组或者列表，包含每个样本的标签ID
for feature in face_features:
    feature = feature.numpy()
face_features = np.vstack(face_features)

print(face_features.shape)

# 初始化t-SNE
tsne = TSNE(n_components=2, random_state=0)

# 进行t-SNE降维
reduced_features = tsne.fit_transform(face_features)
# save reduced features
face_features_dict["reduced_feature"] = reduced_features
with open(feature_path, 'wb') as f:
    pickle.dump(face_features_dict, f)
    
# 绘制t-SNE散点图
plt.figure(figsize=(5, 5))

# 为每个标签画散点图
unique_labels = np.unique(labels)
unique_labels = unique_labels[:20]
for label in unique_labels:
    # 找出当前标签的所有点
    indices = labels == label
    current_feature = reduced_features[indices]
    
    # 绘制属于该标签的点
    plt.scatter(current_feature[:, 0], current_feature[:, 1], label=label)

plt.legend() # 显示图例
plt.title("t-SNE Visualization of Face Features")
# plt.xlabel("t-SNE feature 1")
# plt.ylabel("t-SNE feature 2")
# plt.show() # 显示图
plt.savefig("Features.pdf") # 保存图