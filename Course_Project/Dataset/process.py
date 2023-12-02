import argparse
import os
import shutil
from xml.dom.minidom import parse
from tqdm import tqdm
import random


def create_labels(root_node, labels_root, img_root):
    img_w = 960
    img_h = 540

    frames = root_node.getElementsByTagName('frame')
    for frame in frames:
        seq_gt = []
        frame_num = frame.getAttribute("num")
        frame_num = frame_num.zfill(5)
        label_name = root_node.getAttribute("name") + '_img' + frame_num

        sample_img = os.listdir(img_root)
        if (label_name + '.jpg') in sample_img:
            label_path = os.path.join(labels_root, label_name + '.txt')

            targets = frame.getElementsByTagName('target_list')[0].getElementsByTagName('target')
            for target in targets:
                box = target.getElementsByTagName('box')[0]
                left = float(box.getAttribute('left'))
                top = float(box.getAttribute('top'))
                width = float(box.getAttribute('width'))
                height = float(box.getAttribute('height'))
                type = target.getElementsByTagName('attribute')[0].getAttribute('vehicle_type')
                if type == "car":
                    type = 1
                elif type == "van":
                    type = 2
                elif type == "bus":
                    type = 3
                else:
                    type = 0

                # 中心坐标
                x = left + width / 2
                y = top + height / 2
                # 宽高中心坐标归一化
                x /= img_w
                y /= img_h
                width = width / img_w
                height = height / img_h

                seq_gt.append(str(type) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(
                    round(width, 6)) + ' ' + str(round(height, 6)))

            with open(label_path, 'w') as f:
                for i in seq_gt:
                    f.write(i + '\n')


def yolo_img_train_val(opt):
    # train img
    os.makedirs(opt.new_folder_path+'/images/train')
    os.makedirs(opt.new_folder_path+'/images/val')
    train_images = []  # List to store paths of images for training
    for video_name in tqdm(os.listdir(opt.img_train)):
        if 'MVI' in video_name:
            for img_name in os.listdir(os.path.join(opt.img_train, video_name)):
                if '.jpg' in img_name:
                    ori_path = os.path.join(opt.img_train, video_name, img_name)
                    new_path = os.path.join(opt.new_folder_path+'/images/train/', video_name + '_' + img_name)
                    shutil.copyfile(ori_path, new_path)
                    train_images.append(new_path)

    # Randomly shuffle the list of training images
    random.shuffle(train_images)

    # Select 80% for training
    num_train = int(0.2 * len(train_images))

    # Move the selected images to the training folder
    for img_path in train_images[:num_train]:
        shutil.move(img_path, os.path.join(opt.new_folder_path+'/images/val/', os.path.basename(img_path)))

    print('YOLO training and validation set image has been generated.')


def yolo_img_test(opt):
    os.makedirs(opt.new_folder_path+'/images/test')
    for video_name in tqdm(os.listdir(opt.img_test)):
        if 'MVI' in video_name:
            for img_name in os.listdir(os.path.join(opt.img_test, video_name)):
                if '.jpg' in img_name:
                    ori_path = os.path.join(opt.img_test, video_name, img_name)
                    new_path = os.path.join(opt.new_folder_path+'/images/test/', video_name + '_' + img_name)
                    shutil.copyfile(ori_path, new_path)
    print('YOLO test set image has been generated.')


def gen_yolo_dataset(opt):
    yolo_img_train_val(opt)
    yolo_img_test(opt)

    # train/val label
    os.makedirs(opt.new_folder_path+'/labels/train')
    os.makedirs(opt.new_folder_path+'/labels/val')
    os.makedirs(opt.new_folder_path+'/labels/test')
    for video_xml in tqdm(os.listdir(opt.lbl_train)):
        if '.xml' in video_xml:
            create_labels(parse(os.path.join(opt.lbl_train, video_xml)).documentElement, opt.new_folder_path+'/labels/train/',
                          img_root=opt.new_folder_path+'/images/train/')
            create_labels(parse(os.path.join(opt.lbl_train, video_xml)).documentElement, opt.new_folder_path+'/labels/val/',
                          img_root=opt.new_folder_path+'/images/val/')
    print('YOLO training and validation set label has been generated.')

    # test label
    for video_xml in tqdm(os.listdir(opt.lbl_test)):
        if '.xml' in video_xml:
            create_labels(parse(os.path.join(opt.lbl_test, video_xml)).documentElement, opt.new_folder_path+'/labels/test/',
                          img_root=opt.new_folder_path+'/images/test/')
    print('YOLO test set label has been generated.')

    print('Mission accomplished.')


def parse_opt(new_folder_path, train_path, test_path):
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--new_folder_path', type=str, default=new_folder_path, help='new folder path')
    parser.add_argument('--img_train', type=str, default=train_path, help='train images path')
    parser.add_argument('--img_test', type=str, default=test_path, help='validation images path')
    parser.add_argument('--lbl_train', type=str, default='UA-DETRAC-Annotations/train', help='train labels path')
    parser.add_argument('--lbl_test', type=str, default='UA-DETRAC-Annotations/test', help='validation labels path')
    if os.path.exists(new_folder_path):
        shutil.rmtree(new_folder_path)
    os.makedirs(new_folder_path)
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    yolo_folder_path = 'Test1'
    train_val_path ='UA-DETRAC/gt/train'
    test_path ='UA-DETRAC/gt/test'
    opt = parse_opt(new_folder_path=yolo_folder_path, train_path=train_val_path, test_path=test_path)
    gen_yolo_dataset(opt)
