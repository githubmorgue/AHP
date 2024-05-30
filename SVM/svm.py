import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# 获取当前目录路径
current_directory = os.getcwd()

# 构建feature2/AE文件夹路径
feature2_directory = os.path.join(current_directory, "feature2/AE")

# 存储读取的数据的列表
data = []

# 遍历feature2/AE文件夹中的所有CSV文件
for filename in os.listdir(feature2_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(feature2_directory, filename)
        # 读取CSV文件，并将其转换为NumPy数组
        df = pd.read_csv(file_path, encoding="utf-8")
        data.append(df)

        # 对当前文件进行计算
        x = df.iloc[:20, 1:6].values
        y = df.iloc[:20, 0].values


        # 创建SVM分类器并拟合数据
        clf = SVC(kernel="linear")
        clf.fit(x, y)

        # 使用第30行的第2列到第6列进行预测
        prediction_data = df.iloc[29, 1:6].values
        prediction = clf.predict(prediction_data.reshape(1, -1))

        print("File:", filename)
        print("Prediction:", prediction)
        print("--------------------")