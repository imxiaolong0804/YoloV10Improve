# aug_images_path = r"D:\devProject\detect\yolov10\datasets\GISDATA\AugImg\imageAug2"
# aug_labels_path = r"D:\devProject\detect\yolov10\datasets\GISDATA\AugImg\labelAug2"
#
# images_path = r"D:\devProject\detect\yolov10\datasets\GISDATA\valid\images"
# labels_path = r"D:\devProject\detect\yolov10\datasets\GISDATA\valid\labels"

import os
import cv2
import glob
import albumentations as A

# 设置图像和标签的路径
images_path = r"D:\devProject\detect\yolov10\datasets\GISDATA\valid\images"
labels_path = r"D:\devProject\detect\yolov10\datasets\GISDATA\valid\labels"

# 定义增强管道
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # 水平翻转
    A.RandomBrightnessContrast(p=0.2),  # 随机亮度对比度
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0)  # 随机旋转 ±15度
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.0,  # 设置最小可见性
    # check_each_transform=False      # 如果有此参数，可以尝试设置为 False
))

# 获取所有图像文件列表
image_files = glob.glob(os.path.join(images_path, '*.jpg'))  # 根据实际情况调整扩展名

# 创建用于保存增强后数据的目录
aug_images_path = r"D:\devProject\detect\yolov10\datasets\GISDATA\AugImg\imageAug2"
aug_labels_path = r"D:\devProject\detect\yolov10\datasets\GISDATA\AugImg\labelAug2"

os.makedirs(aug_images_path, exist_ok=True)
os.makedirs(aug_labels_path, exist_ok=True)

for img_file in image_files:
    # 读取图像
    img = cv2.imread(img_file)
    height, width = img.shape[:2]

    # 读取对应的标签文件
    base_name = os.path.splitext(os.path.basename(img_file))[0]
    label_file = os.path.join(labels_path, base_name + '.txt')

    # 加载边界框和类别标签
    bboxes = []
    class_labels = []
    with open(label_file, 'r') as f:
        for line in f:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, bbox_width, bbox_height])
            class_labels.append(int(class_id))

    try:
        # 应用增强
        augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)

        # 手动裁剪边界框
        clipped_bboxes = []
        clipped_class_labels = []
        for bbox, cls_label in zip(augmented['bboxes'], augmented['class_labels']):
            x_center, y_center, bbox_width, bbox_height = bbox

            # 计算边界框的左上角和右下角坐标
            x_min = x_center - bbox_width / 2
            y_min = y_center - bbox_height / 2
            x_max = x_center + bbox_width / 2
            y_max = y_center + bbox_height / 2

            # 裁剪坐标到 [0.0, 1.0]
            x_min = max(0.0, min(1.0, x_min))
            y_min = max(0.0, min(1.0, y_min))
            x_max = max(0.0, min(1.0, x_max))
            y_max = max(0.0, min(1.0, y_max))

            # 重新计算裁剪后的边界框中心坐标和宽高
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            x_center = x_min + bbox_width / 2
            y_center = y_min + bbox_height / 2

            # 过滤掉无效的边界框（宽或高小于等于0）
            if bbox_width <= 0 or bbox_height <= 0:
                continue

            # 将裁剪后的边界框添加到列表中
            clipped_bboxes.append([x_center, y_center, bbox_width, bbox_height])
            clipped_class_labels.append(cls_label)

        # 如果没有有效的边界框，跳过此图像
        if len(clipped_bboxes) == 0:
            continue

        # 保存增强后的图像
        aug_img_name = os.path.join(aug_images_path, base_name + '_aug.jpg')
        cv2.imwrite(aug_img_name, augmented['image'])

        # 保存增强后的标签
        aug_label_name = os.path.join(aug_labels_path, base_name + '_aug.txt')
        with open(aug_label_name, 'w') as f:
            for bbox, class_id in zip(clipped_bboxes, clipped_class_labels):
                x_center, y_center, bbox_width, bbox_height = bbox
                f.write(f'{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n')

    except Exception as e:
        print(f'处理图像 {img_file} 时出错：{e}')
        continue
