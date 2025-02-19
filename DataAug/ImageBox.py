import os
import cv2
import glob
import random

# 设置图像和标签的路径
images_path = r"D:\devProject\detect\new_images\X.v5i.yolov8\test\images"  # 原始图像目录
labels_path = r"D:\devProject\detect\new_images\X.v5i.yolov8\test\labels"  # 标签文件目录
output_path = r"D:\devProject\detect\new_images\box"  # 带有边界框的图像输出目录

# 创建输出目录
os.makedirs(output_path, exist_ok=True)

colors = {}

class_names = {0: 'bubble', 1: 'crack', 2: 'foreign_body', 3: 'malposition'}

# 获取所有图像文件列表
image_files = glob.glob(os.path.join(images_path, '*.jpg'))  # 根据实际情况调整扩展名

for img_file in image_files:
    # 读取图像
    img = cv2.imread(img_file)
    height, width = img.shape[:2]

    # 读取对应的标签文件
    base_name = os.path.splitext(os.path.basename(img_file))[0]
    label_file = os.path.join(labels_path, base_name + '.txt')

    # 检查标签文件是否存在
    if not os.path.exists(label_file):
        print(f'标签文件不存在：{label_file}')
        continue

    # 加载边界框和类别标签
    bboxes = []
    class_labels = []
    with open(label_file, 'r') as f:
        for line in f:
            # 解析标签文件中的每一行
            items = line.strip().split()
            if len(items) != 5:
                print(f'标签格式错误：{label_file}')
                continue
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, items)
            class_labels.append(int(class_id))

            # 将 YOLO 格式转换为边界框的左上角和右下角坐标
            x_min = int((x_center - bbox_width / 2) * width)
            y_min = int((y_center - bbox_height / 2) * height)
            x_max = int((x_center + bbox_width / 2) * width)
            y_max = int((y_center + bbox_height / 2) * height)
            bboxes.append([x_min, y_min, x_max, y_max])

    # 为新的类别生成随机颜色
    for class_id in class_labels:
        if class_id not in colors:
            colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # 绘制边界框
    for bbox, class_id in zip(bboxes, class_labels):
        x_min, y_min, x_max, y_max = bbox

        # 获取对应类别的颜色
        color = colors[class_id]

        # 绘制矩形边框
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=5)

        # 在边框上方显示类别 ID
        label = class_names.get(class_id, 'Unknown')
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, thickness=3)

    # 保存带有边界框的图像
    output_file = os.path.join(output_path, base_name + '_annotated.jpg')
    cv2.imwrite(output_file, img)
    print(f'已保存标注图像：{output_file}')
