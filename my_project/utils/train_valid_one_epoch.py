# --coding:utf-8--
import torch
from tqdm import tqdm
import sys


def train_one_epoch(model, optimizer, loss_f, data_loader, device, epoch):
    model.train()  # 训练模式

    loss_function = loss_f
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = torch.Tensor(images).to(device)
        labels = torch.Tensor(labels).to(device)

        sample_num += images.shape[0]

        pred = model(images)  # 最后预测的结果，各类的置信度
        # _, pred_classes = torch.max(pred, dim=1) # 与后面这句等价
        pred_classes = torch.max(pred, dim=1)[1]  # 求出置信度最大的那个索引

        accu_num += torch.eq(pred_classes, labels).sum()  # 累计正确数量

        loss = loss_function(pred, labels)  # 计算损失
        # loss = torch.nn.CrossEntropyLoss(pred, labels)
        loss.backward()  # 反向传播
        accu_loss += loss.detach()  # 把requires_grad=False梯度去掉不能进行梯度下降

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):  # 如果loss无界，即正无穷大，则停止训练
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, loss_f, data_loader, device, epoch):
    loss_function = loss_f

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = torch.Tensor(images).to(device)
        labels = torch.Tensor(labels).to(device)
        sample_num += images.shape[0]

        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
