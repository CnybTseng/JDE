# JDE
Towards Real-Time Multi-Object Tracking <br>

The code in the mot directory is easier to use when training. Use tools/train.py instead of ./train.py. The *.py file in the root directory will be deprecated.

<del># 1. 训练步骤<del>

<del>以下操作步骤均以MOT16为例</del>

<del>## 1.1 准备数据集</del>

<del>* 从MOT挑战赛官网下载数据集并解压 <br></del>
<del>wget https://motchallenge.net/data/MOT16.zip -P /data/tseng/dataset/jde <br></del>
<del>cd /data/tseng/dataset/jde <br></del>
<del>unzip MOT16.zip -d MOT16 <br></del>

<del>* 创建MOT16任务的工作区, 并将MOT格式标注文件转换为需要格式的标注文件 <br></del>
<del>git clone https://github.com/CnybTseng/JDE.git <br></del>
<del>cd JDE <br></del>
<del>mkdir -p workspace/mot16-2020-5-29 <br></del>
<del>./tools/split_dataset.sh ./workspace/mot16-2020-5-29 <br></del>
<del>此时workspace/mot16-2020-5-29目录下会生成train.txt <br></del>

<del>## 1.2 从预训练模型导出参数生成JDE初始模型<del>

<del>* 从darknet官网下载darknet53预训练模型 <br><del>
<del>wget https://pjreddie.com/media/files/darknet53.conv.74 -P ./workspace <br><del>
<del>python darknet2pytorch.py -pm ./workspace/mot16-2020-5-29/jde.pth \ <br><del>
<del>    --dataset ./workspace/mot16-2020-5-29 -dm ./workspace/darknet53.conv.74 -lbo <br><del>
<del>此时workspace/mot16-2020-5-29目录下会生成初始模型jde.pth, 其骨干网已初始化为darnet53的参数 <br><del>

<del>## 1.3 执行训练脚本<del>

<del>cp ./tools/train.sh ./train.sh <br><del>
<del>根据需要修改, 然后运行训练脚本 <br><del>
<del>./train.sh <br><del>

<del># 2. 测试<del>
<del>运行类似如下命令执行多目标跟踪 <br><del>
<del>python tracker.py --img-path /data/tseng/dataset/jde/MOT16/test/MOT16-03/img1 \ <br><del>
<del>    --model workspace/mot16-2020-5-29/checkpoint/jde-ckpt-049.pth<del>