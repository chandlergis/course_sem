import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time

# 初始化客户端，使用环境变量获取API密钥
# 初始化客户端，直接传入API密钥
client = OpenAI(
    api_key="352d25f8-1a02-483c-8b74-b50e6761d3ff",  # 直接写在这里
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 读取Excel文件
file_path = r"D:\github\xiaoduo\semenic_anylsis\project_name.xlsx"
df = pd.read_excel(file_path)
ids = df.iloc[:, 0].tolist()  # 第一列是ID
project_names = df.iloc[:, 1].tolist()  # 第二列是项目名称

# 定义函数调用API生成嵌入
def get_embeddings(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {len(texts) // batch_size + 1}")
        try:
            resp = client.embeddings.create(
                model="doubao-embedding-text-240515",
                input=batch,
                encoding_format="float"
            )
            batch_embeddings = [item.embedding for item in resp.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error in batch {i}: {e}")
        time.sleep(1)  # 避免触发API频率限制
    return embeddings

# 获取所有项目名称的嵌入
print("----- Starting embeddings request -----")
embeddings = get_embeddings(project_names)
embeddings = np.array(embeddings)  # 转为numpy数组

# 计算相似度矩阵
print("----- Calculating similarity matrix -----")
similarity_matrix = cosine_similarity(embeddings)

# 收集重复的项目对
threshold = 0.9
duplicates = []
print("----- Finding duplicates -----")
for i in range(len(project_names)):
    for j in range(i + 1, len(project_names)):
        if similarity_matrix[i][j] > threshold:
            duplicate_entry = {
                "ID_1": ids[i],
                "Project_Name_1": project_names[i],
                "ID_2": ids[j],
                "Project_Name_2": project_names[j],
                "Similarity": similarity_matrix[i][j]
            }
            duplicates.append(duplicate_entry)
            print(f"重复: ID={ids[i]} ({project_names[i]}) 和 ID={ids[j]} ({project_names[j]}) "
                  f"(相似度: {similarity_matrix[i][j]:.3f})")

# 导出到Excel
if duplicates:
    output_df = pd.DataFrame(duplicates)
    output_path = r"D:\github\xiaoduo\semenic_anylsis\duplicate_projects.xlsx"
    output_df.to_excel(output_path, index=False)
    print(f"----- Results exported to {output_path} -----")
else:
    print("----- No duplicates found -----")

print("----- Done -----")