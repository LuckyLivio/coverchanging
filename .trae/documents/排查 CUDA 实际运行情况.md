# 添加 Web 前端演示与项目配置优化

## 1. 配置优化
*   **更新 `.gitignore`**: 追加以下规则以忽略大文件和生成文件：
    ```gitignore
    # Project specific
    data/
    results/
    models/*.pth
    ```
*   **更新 `requirements.txt`**: 添加 `gradio`。

## 2. Web 演示应用 (`app.py`)
创建一个基于 Gradio 的交互式脚本 `app.py`，功能如下：
1.  **模型初始化**: 自动查找并加载 `models/best_model.pth`。
2.  **推理管道**:
    *   `preprocess(img)`: Resize (256x256) -> ToTensor -> Normalize。
    *   `predict(t1, t2)`: 执行模型前向传播 -> Sigmoid -> Threshold (0.5)。
    *   `postprocess(mask)`: 转换回可视化的 PIL Image。
3.  **可视化**: 生成 "Overlay" 视图（在 T2 图像上叠加红色半透明 Mask）。
4.  **界面布局**:
    *   **Input**: 两个 Image 组件 (Time 1, Time 2)。
    *   **Output**: 两个 Image 组件 (Change Mask, Overlay Result)。

## 3. 文档更新
*   在 `README.md` 中新增章节，指导用户如何启动 Web Demo。