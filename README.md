# JDE
Towards Real-Time Multi-Object Tracking

# 1. 训练步骤

以下操作步骤均以MOT16为例

## 1.1 准备数据集

从MOT挑战赛官网下载数据集并解压
wget https://motchallenge.net/data/MOT16.zip -P /data/tseng/dataset/jde
cd /data/tseng/dataset/jde
unzip MOT16.zip -d MOT16

创建MOT16任务的工作区, 并将MOT格式标注文件转换为需要格式的标注文件
cd $(JDE)
mkdir -p workspace/mot16-2020-5-29
./tools/split_dataset.sh ./workspace/mot16-2020-5-29

此时workspace/mot16-2020-5-29目录下会生成train.txt

## 1.2 从预训练模型导出参数生成JDE初始模型

从darknet官网下载darknet53预训练模型
wget https://pjreddie.com/media/files/darknet53.conv.74 -P ./workspace

python darknet2pytorch.py -pm ./workspace/mot16-2020-5-29/jde.pth \
    --dataset ./workspace/mot16-2020-5-29 -dm ./workspace/darknet53.conv.74 -lbo

此时workspace/mot16-2020-5-29目录下会生成初始模型jde.pth, 其骨干网已初始化为darnet53的参数

## 1.3 执行训练脚本

cp ./tools/train.sh ./train.sh
根据需要修改, 然后运行训练脚本
./train.sh
