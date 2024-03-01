# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:55:22 2024

@author: 86158
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

#k-means算法函数
def k_means(data, k, max_iterations=100):
    # 随机选择k个数据点作为初始聚类中心
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]
    
    for _ in range(max_iterations):
        # 计算每个数据点到聚类中心的距离
        distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
        
        # 将每个数据点分配到距离最近的聚类中心
        labels = np.argmin(distances, axis=0)
        
        # 更新聚类中心为每个簇的平均值
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # 如果聚类中心没有明显变化，则停止迭代
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids



#绘制聚类结果图形
def plot_kmeans_result(data, labels, centroids):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('K-means Clustering Result')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()




#对iris数据集进行聚类
def iris_fun():
    # 加载iris数据集
    iris = load_iris()
    data = iris.data

    # 调用K-means算法
    k = 3
    labels, centroids = k_means(data, k)

    # 绘制聚类结果图形
    plot_kmeans_result(data, labels, centroids)
    
    # 计算轮廓系数
    print("silhouette_score： ",silhouette_score(data, labels))
    
    
#对boston数据集进行聚类
def boston_fun():
    # 加载boston数据集
    boston = load_boston()
    data = boston.data

    # 调用K-means算法
    k = 5
    labels, centroids = k_means(data, k)

    # 绘制聚类结果图形
    plot_kmeans_result(data, labels, centroids)
    
    # 计算轮廓系数
    print("silhouette_score： ",silhouette_score(data, labels))
    
    
#对diabetes数据集进行聚类
def diabetes_fun():
    # 加载diabetes数据集
    diabetes = load_diabetes()
    data = diabetes.data

    # 调用K-means算法
    k = 2
    labels, centroids = k_means(data, k)

    # 绘制聚类结果图形
    plot_kmeans_result(data, labels, centroids)
    
    # 计算轮廓系数
    print("silhouette_score： ",silhouette_score(data, labels))
    
    
#对wine数据集进行聚类
def wine_fun():
    # 加载wine数据集
    wine = load_wine()
    data = wine.data

    # 调用K-means算法
    k = 3
    labels, centroids = k_means(data, k)

    # 绘制聚类结果图形
    plot_kmeans_result(data, labels, centroids)
    
    # 计算轮廓系数
    print("silhouette_score： ",silhouette_score(data, labels))
    
    
#对breast_cancer数据集进行聚类
def breast_cancer_fun():
    # 加载breast_cancer数据集
    breast_cancer = load_breast_cancer()
    data = breast_cancer.data

    # 调用K-means算法
    k = 2
    labels, centroids = k_means(data, k)

    # 绘制聚类结果图形
    plot_kmeans_result(data, labels, centroids)
    
    # 计算轮廓系数
    print("silhouette_score： ",silhouette_score(data, labels))

if __name__ == '__main__':
    iris_fun()
    boston_fun()
    diabetes_fun()
    wine_fun()
    breast_cancer_fun()