import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join


def convert(size, box):
    # size=(width, height)  b=(xmin, xmax, ymin, ymax)
    # x_center = (xmax+xmin)/2        y_center = (ymax+ymin)/2
    # x = x_center / width            y = y_center / height
    # w = (xmax-xmin) / width         h = (ymax-ymin) / height

    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]

    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]

    # print(x, y, w, h)
    return (x, y, w, h)


def convert_annotation(xml_files_path, save_txt_files_path, classes):
    xml_files = os.listdir(xml_files_path)
    # print(xml_files)
    for xml_name in xml_files:
        print(xml_name)
        xml_file = os.path.join(xml_files_path, xml_name)
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')
        out_txt_f = open(out_txt_path, 'w')
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)


        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            # b=(xmin, xmax, ymin, ymax)
            # print(w, h, b)
            bb = convert((w, h), b)
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# 转换xml文件
def bboxes2xml(folder, img_name, width, height, gts, xml_save_to):
    xml_file = open((xml_save_to + '/' + img_name + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>' + folder + '</folder>\n')
    xml_file.write('    <filename>' + str(img_name) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    for gt in gts:
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(gt[0]) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(gt[1]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(gt[2]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(gt[3]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(gt[4]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')
    xml_file.close()


# # DETRAC-Train-Annotations-XML---内涵xml文件
# xmlDir= "Dataset/UA-DETRAC-Annotations/train"
# # 转换的xml文件保存位置
# new_dir="Dataset/UA-DETRAC-Annotations-Yolo/train"
#
#
# for xmlNames in os.listdir(xmlDir):
#     xmlPath=os.path.join(xmlDir,xmlNames)   # xml文件路径
#     # print(xmlPath)
#     tree = ET.parse(xmlPath)
#     root = tree.getroot()
#     findall_frames = root.findall("frame")            # frame标签列表
#     fileName=root.attrib["name"]
#     # print(fileName)
#
#
#     for findall_frame in findall_frames:
#         attrib = findall_frame.attrib["num"]
#         zfill = attrib.zfill(5)
#         imageName="img"+zfill        # 图像的名称
#         print("--------------------------{}".format(imageName))
#         gts = []
#         target_list = findall_frame.findall("target_list")[0]
#
#         findall_targets = target_list.findall("target")      # target对应的标签
#         for findall_target in findall_targets:
#             gt_temp = []
#             LabelName = findall_target.findall("attribute")[0].attrib["vehicle_type"]       # 获取标签类别
#             gt_temp.append(LabelName)
#             box_Dict = findall_target.findall("box")[0].attrib   # 标注物体坐标
#             xmin = float(box_Dict["left"])
#             ymin = float(box_Dict["top"])
#             width = float(box_Dict["width"])
#             height = float(box_Dict["height"])
#             xmax=xmin+width
#             ymax=ymin+height
#             gt_temp.append(int(xmin))
#             gt_temp.append(int(ymin))
#             gt_temp.append(int(xmax))
#             gt_temp.append(int(ymax))
#             gts.append(gt_temp)
#
#         print(gts)
#         folder = "images"
#         img_name = fileName+imageName
#         width = 960                         # 图像的像素
#         height = 540                        # 图像像素
#         xml_save_to = new_dir               # 生成的xml保存位置
#
#         # 生成xml文件
#         bboxes2xml(folder, img_name, width, height, gts, xml_save_to)
#         print("done")
#
#         print("----------------------------------------")


if __name__ == "__main__":
    # 把forklift_pallet的voc的xml标签文件转化为yolo的txt标签文件
    # 1、需要转化的类别
    classes = ['car',  'bus',  'van', 'others']  # 注意：这里根据自己的类别名称及种类自行更改
    # 2、voc格式的xml标签文件路径
    xml_files1 = "Dataset/UA-DETRAC-Annotations/test/"
    # 3、转化为yolo格式的txt标签文件存储路径
    save_txt_files1 = "Dataset/UA-DETRAC-Annotations-Yolo/test"

    convert_annotation(xml_files1, save_txt_files1, classes)