# 使用说明

### Terminal常见使用方法

服务器相关：

```bash
# 连接服务器
ssh wzf@162.105.98.86
# 切换conda环境
conda activate opensim-rl # 服务器
# 启动tmux
tmux
# 打开tmux的某窗口，一般是0
tmux attach-session -t 0
```

预处理数据：

```bash
# 给予权限
chmod 777 preprocess_script.sh
# 运行脚本
./preprocess_script.sh 1 128
```

训练程序：

```bash
# 使用GPU训练
python main.py --gpu
```

打开TensorBoard（可视化）：

```bash
cd v1_runs/ # 或 cd v2_runs/
tensorboard --logdir . --port 10100
# 访问网站162.105.98.86:10100(端口号与上边对应)
```

### 各文件用途及使用方式

- compute_statistic.py - 统计每个类别数据的数量、最大序列长度、最小序列长度、平均序列长度，使用matplotlib进行统计数据可视化；使用方法：`python compute_statistic.py`
- **main.py** - 训练主程序，可以根据参数调整超参数；使用方法：`（使用GPU）python main.py --gpu`或`（使用CPU、调整一些参数）python main.py --learning-rate 1e-3 --batch-size 128`
- model.py - HCN网络模型，基于PyTorch的nn.Module
- **preprocess_script.sh** - 自动化预处理骨骼数据脚本，可以自定义数据集版本和序列长度；使用方法：首先`chmod 777 preprocess_script.sh（给予权限）`，`（v1, 128）./preprocess_script.sh 1 128`或`（v2, 64）./preprocess_script.sh 2 64`
- preprocess_skeleton.py - 预处理骨骼数据；使用方法：`python preprocess_skeleton.py ./PKUMMDv1/ 128`
- split_dataset.py - 将数据集划分为train、val、test，用于训练；使用方法：`python split_dataset.py`
- utils.py - 辅助工具函数