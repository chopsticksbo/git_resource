# --coding:utf-8--
import nni
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from utils.F3_mydataset import MyDataset2
from utils.F22_generate_imagePath_label import read_split_data
from utils.train_valid_one_epoch import train_one_epoch, evaluate
from utils.confusion_matrix import validate, show_confMat
from model.My_model import Net

# --------------前期准备：1.划分训练，验证 ，测试集，2.生成对应的含有文件标签的txt文档---------------------

# 获取训练集、验证集的数据信息，方便后面构建实例
data_path = 'D:/XJND/dataset/butterfly/leedsbutterfly/butterfly_img'
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

# ----------------------------------------------指定参数-----------------------------------------------

# 指定设备
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print('device:{} is ready for working.'.format(device))

# 超参数
params = {
    'lr': 0.001,
    'momentum': 0,
    'features': 128
}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

batch_size = 128
max_epoch = 20

classes_name = [
    "Danaus_plexippus", "Heliconius_charitonius", "Heliconius_erato", "Junonia_coenia", "Lycaena_phlaeas",
    "Nymphalis_antiopa", "Papilio_cresphontes", "Pieris_rapae", "Vanessa_atalanta", "Vanessa_cardui"
]

# log
result_dir = os.path.join(".", "Result")
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 状态码
pre_weight = True  # 是否使用迁移学习进行权重进行初始化，True：使用，False：进行模型自带初始化
weights_dir = 'D:\XJND\lufei\pytorch\T001_my_project\weights\model-vgg11_add_last_linear-18.pth'

nni_trail = False  # 是否进行nni超参数实验
trail_num = 1

elimination_parameter = False  # 是否剔除部分层参数
modify_layers = True

# -------------------------------------------- step 1/5 : 加载数据 -------------------------------------------

# 数据预处理设置
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
])
validTransform = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset2实例---->train2_file_string对应MyDataset2
train_data = MyDataset2(images_path=train_images_path, images_class=train_images_label, transform=trainTransform)
valid_data = MyDataset2(images_path=val_images_path, images_class=val_images_label, transform=validTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, )
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size)

# ------------------------------------ step 2/5 : 定义网络 ------------------------------------

# net = Net(num_classes=10, n_features=params['features']).to(device=device)  # 创建一个网络
# 使用 VGG13
net = torchvision.models.vgg11()

# ================================ #
#        finetune 权值初始化
# ================================ #

if pre_weight:  # 用预训练的权值

    # load params
    pretrained_dict = torch.load(weights_dir)

    # 获取当前网络的dict
    net_state_dict = net.state_dict()

    # 剔除不匹配的权值参数
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}

    # 更新新模型参数字典
    net_state_dict.update(pretrained_dict_1)

    # 将包含预训练模型参数的字典"放"到新模型中
    net.load_state_dict(net_state_dict)

else:
    net.initialize_weights()  # 初始化权值

# 是否需要修改部分层或者增加层
if modify_layers:
    # net.add_module('add-linear', nn.Linear(1000, 10))  # 增加一层保证输出的类别一样
    # net.classifier[6] = nn.Linear(4096,10)  # 修改模型的最后一层变成10分类
    net.classifier.add_module('add-linear', nn.Linear(1000, 10))  # 加到VGG的classifier里面
    net.to(device)
else:
    net.to(device)
print(net)
# print(net.parameters())

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

# ==================================
# 是否剔除部分层参数
# 将fc3层的参数从原始网络参数中剔除
# ==================================
if elimination_parameter == True:
    ignored_params = list(map(id, net.fc3.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    # 为fc3层设置需要的学习率
    optimizer = torch.optim.SGD([
        {'params': base_params},  # 前面的层
        {'params': net.fc3.parameters(), 'lr': params['lr'] * 10}],  # 最后一层，进行拼接，并设置专门的学习率
        lr=params['lr'], momentum=params['momentum'], weight_decay=1e-4)

else:
    # optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'], momentum=params['momentum'], dampening=0.1)  # 选择优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'], momentum=params['momentum'])

criterion = nn.CrossEntropyLoss()  # 选择损失函数
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------
best_acc_epoch = 0
for epoch in range(max_epoch):
    # train
    tb_writer = SummaryWriter()
    train_loss, train_acc = train_one_epoch(model=net,
                                            optimizer=optimizer,
                                            loss_f=criterion,
                                            data_loader=train_loader,
                                            device=device,
                                            epoch=epoch)

    scheduler.step()

    # validate
    val_loss, val_acc = evaluate(model=net,
                                 loss_f=criterion,
                                 data_loader=valid_loader,
                                 device=device,
                                 epoch=epoch)

    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    tb_writer.add_scalar(tags[0], train_loss, epoch)
    tb_writer.add_scalar(tags[1], train_acc, epoch)
    tb_writer.add_scalar(tags[2], val_loss, epoch)
    tb_writer.add_scalar(tags[3], val_acc, epoch)
    tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
    best_acc = 0.0
    if val_acc >= best_acc:  # 验证集验证的时候精度大于之前最好的精度则保存其模型,最后的那个模型精度最高
        best_acc = val_acc
        best_acc_epoch = epoch
        torch.save(net.state_dict(), "./weights/model-vgg11_add_last_linear-{}.pth".format(epoch))

        # os.remove('./weights/model-vgg11_add_last_linear-{}.pth'.format()) # 加一个准确率大于之前的模型，就删除上一个模型

    # 此处验证集用来验证模型的好坏，验证集的精度越高，模型就相对较好
    nni.report_intermediate_result(val_acc)
nni.report_final_result(val_acc)
print('Finished Training')

# ------------------------------------ step 5/5 : 绘制混淆矩阵图 ------------------------------------

# 最后用测试集去验证模型的精度
print(best_acc_epoch)
# 加载best_acc_epoch的模型
net.load_state_dict(torch.load('./weights/model-vgg11_add_last_linear-{}.pth'.format(best_acc_epoch)))
conf_mat_train, train_acc = validate(net, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(net, valid_loader, 'valid', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
