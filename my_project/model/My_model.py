# --coding:utf-8--
import torch.nn as nn
import torch.nn.functional as F


# from T001_my_project.train2_file_string import params


class Net(nn.Module):
    def __init__(self, num_classes=10, n_features=128):
        super(Net, self).__init__()
        self.num_class = num_classes
        self.n_features = n_features
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc1 = nn.Linear(16 * 5 * 5, self.n_features)
        self.fc2 = nn.Linear(self.n_features, self.n_features)
        self.fc3 = nn.Linear(self.n_features, self.num_class)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)  # xavier针对饱和激活函数，tanh,sigmoid,kaiming则针对不饱和激活函数,relu,leakrelu
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):  # BatchNorm/LayerNorm weight:1,bias:0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28 * 28, params['features']),
#             nn.ReLU(),
#             nn.Linear(params['features'], params['features']),
#             nn.ReLU(),
#             nn.Linear(params['features'], 10)
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
