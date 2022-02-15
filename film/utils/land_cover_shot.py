import os
import json

import torch
from PIL import Image
from prettytable import PrettyTable
from torch import nn
from torchvision import transforms

from film import resnet
import json


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = r'D:\FILM_Network\gid\batch_predict\image_tabled_2'
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".tif")]

    # read class_indict
    json_path = './class_indices_gid.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet.resnet34(pretrained=False, film_layer=False)

    # load model weights
    weights_path = r'D:\FILM_Network\film\models\ft3-resnet-trial0.pth'
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 15)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 1  # 每次预测时将多少张图片打包成一个batch
    image_dict={}
    table = PrettyTable()
    table.field_names = ["", "axis", "class", "probs"]
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)
                img_path_axis=img_path.split('image2')
                img_path_axis=img_path_axis[1].split('.')
                img_path_axis=img_path_axis[0].split('_')

                axis='('+img_path_axis[1].__str__()+','+img_path_axis[2].__str__()+')'
            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)

            # 分成五类
            classes_gid_5=['bulit_up','forest','farmland','meadow','water']
            predict_builtup=predict[0:,0]+predict[0:,7]+predict[0:,8]+predict[0:,9]
            predict_forest=predict[0:,1]+predict[0:,13]+predict[0:,14]
            predict_farmland=predict[0:,10]+predict[0:,11]+predict[0:,12]
            predict_water=predict[0:,4]+predict[0:,5]+predict[0:,6]
            predict_meadow=predict[0:,2]+predict[0:,3]
            predict_5 = torch.stack([predict_builtup, predict_forest,predict_farmland,predict_water,predict_meadow])
            predict_5=predict_5.t()

            probs, classes = torch.max(predict_5, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 classes_gid_5[cla.numpy()],
                                                                 pro.numpy()))


                table.add_row([img_path_list[ids * batch_size + idx], axis, classes_gid_5[cla.numpy()], pro.numpy()])

                image_dict[axis]=classes_gid_5[cla.numpy()]


        print(image_dict)

        jsObj = json.dumps(image_dict, indent=4)  # indent参数是换行和缩进

        fileObject = open('image2_dict_ft3.json', 'w')
        fileObject.write(jsObj)
        fileObject.close()

    o_dict_path = r'D:\FILM_Network\film\results\json\gid2_konwn_dict.json'
    o_dict = json.load(open(o_dict_path))
    t_dict = image_dict

    for i in (o_dict):
        if o_dict[i] == 'undefined':
            t_dict[i] = 'undefined'

    jsObj = json.dumps(t_dict, indent=4)  # indent参数是换行和缩进

    fileObject = open('image2_join_dict_ft3.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()



if __name__ == '__main__':
    main()
