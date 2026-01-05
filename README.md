# 基于多时相遥感图像的地表变化检测方法研究 (Project Report)

## 1. 项目概述 (Project Overview)

### 1.1 背景
随着遥感技术的快速发展，利用多时相遥感图像进行地表变化检测已成为城市规划、灾害评估和环境监测等领域的重要手段。本项目旨在利用深度学习技术，特别是孪生神经网络（Siamese Network），自动检测同一区域在不同时间点的建筑物变化情况。

### 1.2 目标
1.  理解并实现基于深度学习的变化检测流水线。
2.  构建基于 Siamese U-Net 的变化检测模型。
3.  在 LEVIR-CD 数据集上进行训练与评估，实现自动化变化区域提取。

## 2. 系统与模型设计 (System & Model Design)

### 2.1 整体流程
项目采用端到端的深度学习处理流程：
1.  **数据预处理**: 读取 T1/T2 图像 -> 裁剪/缩放 (256x256) -> 归一化 -> 数据增强。
2.  **特征提取**: 使用共享权重的孪生编码器分别提取 T1 和 T2 的多尺度特征。
3.  **特征融合**: 计算特征的绝对差异 (Difference) 以捕捉变化信息。
4.  **解码与分割**: 通过解码器逐步上采样并融合差异特征，输出像素级的变化概率图。
5.  **结果评估**: 计算 IoU、F1-Score 等指标并可视化。

### 2.2 模型架构: Siamese U-Net
模型代码位于 `models/siamese_net.py`。
*   **Encoder (编码器)**: 采用类 ResNet 结构的卷积模块，包含 4 个下采样阶段。孪生结构意味着 T1 和 T2 分支共享同一套权重，保证特征提取的一致性。
*   **Fusion (融合)**: 采用 `|Feature_T1 - Feature_T2|` 的方式计算差异特征。
*   **Decoder (解码器)**: 采用 U-Net 风格的上采样路径 (Transpose Conv)，并将编码层的差异特征通过 Skip Connection 拼接到解码层，以恢复空间细节。
*   **Output (输出)**: 1x1 卷积 + Sigmoid 激活，输出 (0, 1) 范围的变化概率。

## 3. 实验设置 (Experimental Setup)

### 3.1 数据集
*   **数据集**: LEVIR-CD (建筑物变化检测数据集)。
*   **结构**:
    *   `data/train`: 训练集 (A, B, label)
    *   `data/val`: 验证集
    *   `data/test`: 测试集

### 3.2 训练参数
*   **Loss Function**: BCEWithLogitsLoss (二元交叉熵)。
*   **Optimizer**: AdamW, Learning Rate = 1e-3 (Cosine Annealing)。
*   **Batch Size**: 8 (可根据显存调整)。
*   **Epochs**: 50。

## 4. 快速开始 (Quick Start)

### 4.1 环境安装
```bash
pip install -r requirements.txt
```

### 4.2 数据准备
请下载 LEVIR-CD 数据集，并按以下结构组织到 `data/` 目录：
```text
data/
  train/
    A/ (image files)
    B/ (image files)
    label/ (mask files)
  val/
    ...
  test/
    ...
```

### 4.3 训练模型
```bash
python train.py --epochs 50 --batch_size 8
```
训练过程中会自动保存验证集 F1 分数最高的模型到 `models/best_model.pth`。

### 4.4 预测与可视化
```bash
python predict.py --model_path models/best_model.pth --num_viz 20
```
结果将保存在 `results/` 目录下，包含 T1, T2, GT 和 Prediction 的对比图。

### 4.5 Web 交互演示 (Web Demo)
本项目提供基于 Gradio 的 Web 界面，支持上传任意图片进行测试。
1. 安装依赖: `pip install gradio`
2. 启动服务:
```bash
python app.py
```
3. 打开浏览器访问显示的 URL (通常为 http://127.0.0.1:7860)。
4. 拖入两张图片，点击 "Detect Changes" 即可查看变化区域。

## 5. 结果分析与总结 (Conclusion)

### 5.1 预期结果
*   模型应能有效识别显著的建筑物变化（如新建或拆除）。
*   Siamese 差分结构能够有效抑制由于光照、季节变化引起的伪变化（通过特征提取层的语义抽象能力）。

### 5.2 未来改进
1.  **引入注意力机制**: 如在解码器中加入 Attention Block，增强对边缘细节的关注。
2.  **更换骨干网络**: 使用更强大的 ResNet-50 或 Transformer (如 Swin Transformer) 作为编码器。
3.  **多尺度测试**: 在推理时使用多尺度输入融合以提升精度。

---
*本项目为人工智能课程设计作业，代码基于 PyTorch 实现。*
