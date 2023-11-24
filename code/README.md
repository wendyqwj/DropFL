# HXX && PITY SAY

# federated-learning-dropout-V2.0：相比v1.0，增加了基于统计的异常点检测方法（防护），1）检测出被攻击者妥协的客户端；2）尝试着减弱掉线攻击对模型精度的影响。
# federated-learning-dropout-V1.0：针对FedAvg聚合算法实现了三种掉线攻击（完全掉线、随机掉线和基于SV的掉线），实验评估包括：MNIST、CIFAR10和CIFAR100三个数据集，三个模型MLP、ResNet20和VGG16.



我们魔改了代码, 下面介绍当前工程结构与使用方法:
1. main_fed.py为运行主程序
    使用带参数的指令运行, 如nohup python main_fed.py --num_users 200 --frac 0.1 --alpha 0.3 --delta 0.5 --P 1.0 --gpu 2 --model cnn --dataset cifar100 --MAX_WORKERS 3 > 2.out &
    指令中nohup 与 结尾的 & 表示后台运行, > 2.out 表示将终端输出重定向到此文件.; --MAX_WORKERS 3 代表计算夏普乐值时的最大进程数为3, 此值选取过大会导致GPU out of memory; 
2. /log文件夹下为运行日志, 每次程序运行, 会同步生成以runlog为前缀、以运行信息为文件名的log文件.
3. /save文件夹下为通过log文件数据画出的图, 通过draw_ave.py等脚本绘制.


Note: The scripts will be slow without the implementation of parallel computing. 

## Requirements
python>=3.6  
pytorch>=0.4

## Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

`--all_clients` for averaging over all client models

NB: for CIFAR-10, `num_channels` must be 3.


