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
from utils.confusion_matrix import validate, show_confMat

classes_name = [
    "Danaus_plexippus", "Heliconius_charitonius", "Heliconius_erato", "Junonia_coenia", "Lycaena_phlaeas",
    "Nymphalis_antiopa", "Papilio_cresphontes", "Pieris_rapae", "Vanessa_atalanta", "Vanessa_cardui"
]

data_path = 'D:/XJND/dataset/butterfly/leedsbutterfly/butterfly_img'
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print('device:{} is ready for working.'.format(device))
params = {
    'lr': 0.001,
    'momentum': 0,
    'features': 128
}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

batch_size = 128
max_epoch = 10
# log
result_dir = os.path.join(".", "Result")
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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

net = torchvision.models.vgg11()

conf_mat_train, train_acc = validate(net, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(net, valid_loader, 'valid', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
