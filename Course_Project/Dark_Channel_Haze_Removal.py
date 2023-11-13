#  Reference: https://blog.csdn.net/wsp_1138886114/article/details/95012769
import cv2
import numpy as np
import os


def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                           # 得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照

    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认不进行该操作
    return Y


def dehaze_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all JPG files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".jpg"):
            input_path = input_folder + '/' + file_name
            output_path = output_folder + '/' + file_name

            # Read the image, apply deHaze, and save the result
            dehazed_img = deHaze(cv2.imread(input_path) / 255.0) * 255
            cv2.imwrite(output_path, dehazed_img)


def get_folder_names(parent_folder):
    # Get a list of all items in the parent folder
    items = os.listdir(parent_folder)
    # Filter out only folders
    # folder_names = [parent_folder+'/'+item for item in items]
    return items


if __name__ == '__main__':
    # img_path = "Dataset/UA-DETRAC/hazy/train/MVI_20011_229_0.03/img00386.jpg"
    # m = deHaze(cv2.imread(img_path) / 255.0) * 255
    # cv2.imwrite('result.png', m)

    train_folder_path = "Dataset/UA-DETRAC/hazy/train"
    test_folder_path = "Dataset/UA-DETRAC/hazy/test"
    train_output_folder_path = "Dataset/UA-DETRAC/dehaze_DarkChannel/train"
    test_output_folder_path = "Dataset/UA-DETRAC/dehaze_DarkChannel/test"

    train_folder_list = get_folder_names(train_folder_path)
    test_folder_list = get_folder_names(test_folder_path)
    print(len(train_folder_list), len(test_folder_list))
    for folder_name in train_folder_list:
        input_folder = train_folder_path + '/' + folder_name
        output_folder = train_output_folder_path + '/' + folder_name
        print(input_folder, output_folder)
        dehaze_folder(input_folder, output_folder)

    for folder_name in test_folder_list:
        input_folder = test_folder_path + '/' + folder_name
        output_folder = test_output_folder_path + '/' + folder_name
        print(input_folder, output_folder)
        dehaze_folder(input_folder, output_folder)

