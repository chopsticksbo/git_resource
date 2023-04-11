# --coding:utf-8--
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def validate(net, data_loader, set_name, classes_name):
    """
    对一批数据进行预测，返回混淆矩阵以及Accuracy
    :param net:
    :param data_loader:
    :param set_name:  eg: 'valid' 'train' 'test
    :param classes_name:
    :return:
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    net.eval()
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])

    for step, data in enumerate(data_loader):
        images, labels = data
        images = torch.Tensor(images).to(device)
        labels = torch.Tensor(labels).to(device)

        pre = net(images)
        pre.detach_()

        _, predicted = torch.max(pre, 1)

        # 统计混淆矩阵
        for i in range(len(labels)):
            cate_i = labels[i].cpu().numpy()
            pre_i = predicted[i].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.0

    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
            classes_name[i], np.sum(conf_mat[i, :]), conf_mat[i, i], conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),
                                                                     conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))))

    print('{} set Accuracy:{:.2%}'.format(set_name, np.trace(conf_mat) / np.sum(conf_mat)))

    return conf_mat, '{:.2}'.format(np.trace(conf_mat) / np.sum(conf_mat))


def show_confMat(confusion_mat, classes, set_name, out_dir):
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)
    plt.show()

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    plt.close()
