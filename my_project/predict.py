import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# from vit_model import vit_base_patch16_224_in21k as create_model
from model.My_model import Net

"""
参数修改
1.net = Net(num_classes=10, n_features=128).to(device)  
里面的num_classes改为自己的分类类数，n_features根据超参数调优结果来定
2.img_path = "../Danaus_plexippus.png" 图像路径，改为自己需要识别的图片
3.model_weight_path = "./weights/model-9.pth" 模型路径，精度最高的模型路径
"""


def main():
    data_transform = transforms.Compose(
        [transforms.Resize(50),
         transforms.CenterCrop(32),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = "./Danaus_plexippus.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = Net(num_classes=10, n_features=128).to(device) # 种类数需要修改

    # load model weights
    model_weight_path = "./weights/model-19.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)

        predict_cla = torch.argmax(predict).numpy()
        print('predict_cla:', predict_cla)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)

    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

    print('it maybe:{}'.format(class_indict[str(predict_cla)]))


if __name__ == '__main__':
    main()
