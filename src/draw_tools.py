import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import clear_output
from scipy.interpolate import splprep, splev
import draw_tools
import shutil
import glob
import time

def fix_edge_contour(contour, im_shape):
    """
    有时候生成的轮廓点会有一些头部或者尾部紧挨着图像边沿的情况，这样的点位是不需要的，需要过滤掉。
    如果轮廓点头部或者尾部紧挨着图像边沿，过滤裁掉该部分的点位
    """
    # 将轮廓转换为列表
    contour = contour.tolist()

    # 检查轮廓的头部点
    while True:
        x, y = contour[0][0]
        if x == 0 or y == 0 or x == (im_shape[1] - 1) or y == (im_shape[0] - 1):
            del contour[0]
        else:
            break

    # 检查轮廓的尾部点
    while True:
        x, y = contour[-1][0]
        if x == 0 or y == 0 or x == (im_shape[1] - 1) or y == (im_shape[0] - 1):
            del contour[-1]
        else:
            break

    # 将轮廓转换回numpy数组
    contour = np.array(contour)
    return contour

def getContourList(image, pen_width: int = 3, min_contour_len: int = 30, is_show: bool = False):
    """
    从图像中获取轮廓列表
    :param image: 图像
    :param pen_width: 笔的粗细
    :param min_contour_len: 最短的轮廓长度
    :param is_show: 是否显示图像
    :return: 轮廓列表
    """
    # 读取图片
    # im = cv2.imread("../data/1_fake.png",cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Can't read the image file.")
        return
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)  
    # 转换二值化
    image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

    # 获取图像线条的绘制顺序，以方便于机器人连续运动绘制图像
    # Create a copy of the original image to draw contours on
    image_copy = image.copy()
    image_with_contours = np.full_like(image_copy, 255)

    # Initialize a list to store the contours
    contour_list = []

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    sec0 = (0, image_copy.shape[0])
    sec1 = (sec0[1]-1, sec0[1]+image_copy.shape[1]-1)
    sec2 = (sec1[1]-1, sec1[1]+image_copy.shape[0]-1)
    sec3 = (sec2[1]-1, sec2[1]+image_copy.shape[1]-2)
    while True:
        # Find contours in the image
        # 并且找到的轮廓都在黑色的像素上
        contours, _ = cv2.findContours(image_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # If no contours are found, break the loop
        # 没有轮廓需要中止；当图像是全白时，可以检测到一个轮廓，也需要终止
        if len(contours) == 0 or (len(contours)==1 and np.all(image_copy == 255)):
            break

        # Remove the border contour
        # contours = [cnt for cnt in contours if not np.any(cnt == 0) and not np.any(cnt == height-1) and not np.any(cnt == width-1)]
        # `cv2.findContours`函数在找到轮廓时，实际上是在找到黑色对象（前景）和白色背景之间的边界
        # 这意味着轮廓的坐标可能不会精确地落在原始图像的黑色像素上，而是在黑色和白色像素之间。
        # 如果你希望轮廓精确地落在黑色像素上，需要对`cv2.findContours`的结果进行一些后处理。例如，遍历轮廓的每个点，然后将它们的坐标向最近的黑色像素进行取整。
        # 避免后续在擦除时，并没有擦除原有图像的黑色像素
        print(f"pen width: {pen_width}")
        if pen_width == 1:
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    if image_copy[y, x] == 255:
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if nx >= 0 and ny >= 0 and nx < image_copy.shape[1] and ny < image_copy.shape[0]:
                                if image_copy[ny, nx] == 0:
                                    point[0][0] = nx
                                    point[0][1] = ny
                                    break

        cv2.drawContours(image_with_contours, contours, -1, 0, 1)
        # erase the exist contours
        cv2.drawContours(image_copy, contours, -1, 255, pen_width) 
        # add contours to list
        # Sort the elements in contours according to the length of the elements. 
        # The longest contour is at the front, which is convenient for subsequent drawing and can be drawn first.

        # remove the contour when the contour is the box of image
        contours = list(contours)
        max_len = 0
        for i in reversed(range(len(contours))):
            # 太短的也不要
            if len(contours[i]) < min_contour_len:
                contours.pop(i)
                continue
            # 将画四个边框的轮廓去掉
            if (len(contours[i]) >= ( image_with_contours.shape[0]*2 + image_with_contours.shape[0]*2 - 4) and \
                (contours[i][sec0[0]:sec0[1], :, 0] == 0).all() and \
                (contours[i][sec1[0]:sec1[1], :, 1] == image_with_contours.shape[0]-1).all() and \
                (contours[i][sec2[0]:sec2[1], :, 0] == image_with_contours.shape[1]-1).all() and \
                (contours[i][sec3[0]:sec3[1], :, 1] == 0).all()):
                contours.pop(i)
                continue
        contours.sort(key=lambda x: x.shape[0], reverse=True)
        contour_list.extend(contours)
        if is_show:
            # show the image with the drawn contours
            # Clear the previous plot
            clear_output(wait=True)
            
            plt.subplot(1,3,1)
            plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            
            plt.subplot(1,3,2)
            plt.imshow(image_copy, cmap='gray', vmin=0, vmax=255)

            plt.subplot(1,3,3)
            # Show the image with the current contour
            plt.imshow(image_with_contours, cmap='gray', vmin=0, vmax=255)
            plt.show()
    for i in reversed(range(len(contour_list))):
        contour = contour_list[i]
        contour = fix_edge_contour(contour=contour, im_shape=image.shape)
        if len(contour) < min_contour_len:
            contour_list.pop(i)
    return contour_list


def sortContoursList(contour_list):
    """
    根据以下规则排序：
    1. 先从最长的1/3个轮廓中，挑选出最长的一些轮廓（大致1/5的轮廓）
    2. 以上一个轮廓的终点为准，找到剩下轮廓中，起点与该点位最近的距离排序
    """
    contour_list.sort(key=lambda x: x.shape[0], reverse=True)
    # 数量太少，直接返回排序后的轮廓列表，不需要太多策略
    if len(contour_list) <= 10:
        return contour_list
    origin_count = len(contour_list)
    # 1. 先从最长的1/3个轮廓中，随机选出一些轮廓（大致1/2的轮廓），并从原始轮廓列表中删除
    # 这样画尝的轮廓容易先画出来图像的大体轮廓。另外，随机一下，是为了避免每次都是画同样或者相似的轮廓
    tmp_contour_list = contour_list[:int(len(contour_list)/3)]
    np.random.shuffle(tmp_contour_list)
    tmp_contour_list = tmp_contour_list[:int(len(tmp_contour_list)/2)]
    for contour in tmp_contour_list:
        for i in reversed(range(len(contour_list))):
            if contour_list[i] is contour:
                contour_list.pop(i)
                break
    ret_contour_list = tmp_contour_list
    # 2. 以上一个轮廓的终点为准，找到剩下轮廓中，起点与该点位最近的距离排序
    count = len(tmp_contour_list)
    while (count < origin_count):
        # 找到最后一个轮廓的终点
        last_contour = ret_contour_list[-1]
        last_point = last_contour[-1][0]
        # 找到剩下轮廓中，起点与该点位最近的距离排序
        min_index = -1
        min_distance = 999999999
        for i in range(len(contour_list)):
            first_point = contour_list[i][0][0]
            distance = (first_point[0] - last_point[0])**2 + (first_point[1] - last_point[1])**2
            if distance < min_distance:
                min_distance = distance
                min_index = i
        ret_contour_list.append(contour_list[min_index])
        contour_list.pop(min_index)
        count += 1
    return ret_contour_list

def sample_and_smooth_contours(image, contour_list, interval: int = 5, is_interpolation: bool = True, is_show: bool = False):
      """
      采样并平滑拟合轮廓
      :param image: 图像
      :param contour_list: 轮廓列表
      :param interval: 采样间隔
      :param is_interpolation: 是否进行插值
      :return: 平滑拟合并采样后的轮廓列表。注意为浮点的数组
      """
      # 创建一个新的图片，长宽都增加了30个像素
      if is_show:
          draw_page = np.full((image.shape[0] + 60, image.shape[1] + 60), 255)
      f_contour_list = []
      for contour in contour_list:
          contour = fix_edge_contour(contour=contour, im_shape=image.shape)
          # break
          if (is_interpolation):
              # 对contour中的点进行B样条进行拟合，然后平滑和重采样，
              # Fit a B-spline to the contour
              if (contour[0] == contour[-1]).all():
                  contour = contour.reshape(-1, 2)       
                  tck, u = splprep(contour.T, w=None, u=None, ue=None, k=3, task=0, s=1.0, t=None, full_output=0, nest=None, per=1, quiet=1)
              else:
                  contour = contour.reshape(-1, 2)
                  tck, u = splprep(contour.T, w=None, u=None, ue=None, k=3, task=0, s=1.0, t=None, full_output=0, nest=None, per=0, quiet=1)
              # 设置重采样的点数
              num = contour.shape[0] // interval
              u_new = np.linspace(u.min(), u.max(), num)
              x_new, y_new = splev(u_new, tck, der=0)
              f_contour = np.array([x_new, y_new]).T.reshape(-1, 1, 2)
              f_contour_list.append(f_contour)
          else:
              f_contour = contour.astype(np.float128)
              f_contour_list.append(f_contour)
          # 遍历轮廓中的每个点
          if is_show:          
            for point in f_contour:
                x, y = int(point[0][0]+0.5), int(point[0][1]+0.5)
                # 绘制
                draw_page[y + 30, x + 30] = 0
            clear_output(wait=True)
                # plt.show()
            plt.imshow(draw_page, cmap='gray', vmin=0, vmax=255)
            # 将轮廓中的图像四边的框去掉
            plt.pause(0.001)
      return f_contour_list


def save_contour_points(contour_list, filepath):
  """
  保存轮廓点到文件中，每个轮廓占一行，x和y坐标用逗号分割，点之间用逗号分割
  Usage:
    save_contour_points(f_contour_list, "../data/1_fake_data.txt")
  """
  with open(filepath, "w") as f:
    for contour in contour_list:
      for point in contour:
        x, y = point[0]
        f.write(f"{x},{y},")
      f.write("\n")

def generate_style_image(image):
    plt.imsave("../data/input.jpg", image)
    shutil.copy("../data/input.jpg", "../../QMUPD/examples/input.jpg")
    start_time = time.time()
    curr_path = os.getcwd()
    #================== settings ==================
    style_root = "../../QMUPD/"
    os.chdir(style_root)

    exp = 'QMUPD_model'
    epoch='200'
    dataroot = 'examples'
    gpu_id = '-1'
    netga = 'resnet_style2_9blocks'
    model0_res = 0
    model1_res = 0
    imgsize = 512
    extraflag = ' --netga %s --model0_res %d --model1_res %d' % (netga, model0_res, model1_res)
    vec = [0,1,0]
    svec =  '%d,%d,%d' % (vec[0],vec[1],vec[2])
    img1 = 'output_image'
    command = 'python3 test.py --dataroot %s --name %s --model test --output_nc 1 --no_dropout --model_suffix _A %s --num_test 1000 --epoch %s --style_control 1 --imagefolder %s --sinput svec --svec %s --crop_size %d --load_size %d --gpu_ids %s' % (dataroot,exp,extraflag,epoch,img1,svec,imgsize,imgsize,gpu_id)
    os.system(command)
    print('cost time: ',time.time()-start_time,'s')
    os.chdir(curr_path)
    shutil.copy("../../QMUPD/results/QMUPD_model/test_200/output_image/input_fake.png", "../data/input_fake.png")
    return cv2.imread("../data/input_fake.png",cv2.IMREAD_GRAYSCALE)


# mian
if __name__ == "__main__":
    # 读取图片
    im = cv2.imread("../data/1_fake.png",cv2.IMREAD_GRAYSCALE)
    # 获取轮廓列表
    contour_list = getContourList(im, is_show=True)
    # 对轮廓列表进行排序
    contour_list = sortContoursList(contour_list)
    # 平滑拟合并采样轮廓
    f_contour_list = sample_and_smooth_contours(im, contour_list, is_show=True)
    # 保存轮廓点到文件中，每个轮廓占一行，x和y坐标用逗号分割，点之间用逗号分割
    save_contour_points(f_contour_list, "../data/1_fake_data.txt")