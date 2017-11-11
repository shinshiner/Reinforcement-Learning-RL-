### 一个A3CLSTM的pytorch实现玩俄罗斯方块

## 输入
默认10 * 20, 单通道, 网络结构可在`model.py`里改

## 运行
先运行  `sudo pip3 install -e .` 安装游戏环境
#### PC
`python3 main.py`   (cpu & gpu)

或

`CUDA_VISIBLE_DEVICES=0 python3 main.py`   (gpu)

#### 服务器终端

`sh test.sh`
