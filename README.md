# Live2Diff
Live2Diff：基于点控制视频生成模型和缓存图片策略，模拟live2D效果的模型。

## 安装步骤
```bash
conda create -n live2diff python==3.10.11 -y
conda activate live2diff
git clone https://github.com/Lvshu6/Live2Diff.git
cd Live2Diff
pip install -e .
pip install -r requirements.txt
cd co-tracker
pip install -e .
cd ..

#下载rife补帧模型权重
wget https://hf-mirror.com/Lvshu6/Live2Diff/resolve/main/Live2D/train_log.zip
unzip train_log.zip
```

## 运行示例(Live2D)
```bash
# 从huggingface下载并解压角色包到app/Live2D文件夹下 https://hf-mirror.com/Lvshu6/Live2Diff
cd app/Live2D
wget https://hf-mirror.com/Lvshu6/Live2Diff/resolve/main/Live2D/nuero.zip
unzip nuero.zip
python app_qt5.py
```

## 运行示例(diffusion)
```bash
# 下载模型权重到models/FlowLineAdapter
cd models/FlowLineAdapter
wget https://hf-mirror.com/Lvshu6/Live2Diff/resolve/main/models/FlowLineAdapter/flow_line_adapter.safetensors
cd ../..
# 运行webui(首次生成视频时自动下载模型权重) 
python app/diffusion/app.py
```
