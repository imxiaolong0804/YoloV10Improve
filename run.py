# -*- coding:utf-8 -*-
from ultralytics import YOLOv10
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train or validate YOLO model.')
# train用于训练原始模型  val 用于得到精度指标
parser.add_argument('--mode', type=str, default='train', help='Mode of operation.')
# 预训练模型
# parser.add_argument('--weights', type=str, default='yolov10s.pt', help='Path to model file.') # 训练的
parser.add_argument('--weights', type=str, default='models/yolov10n.pt', help='Path to model file.')  # 测试的
# 数据集存放路径
# parser.add_argument('--data', type=str, default='datasets/VOC2007/data.yaml', help='Path to data file.')
parser.add_argument('--data', type=str, default='datasets/NEU-DET/data.yaml', help='Path to data file.')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs.')
parser.add_argument('--batch', type=int, default=64, help='Batch size.')
parser.add_argument('--workers', type=int, default=0, help='Number of workers.')
parser.add_argument('--device', type=str, default='0', help='Device to use.')
parser.add_argument('--name', type=str, default='test', help='Name data file.')
args = parser.parse_args()


def train(model, data, epoch, batch, workers, device, name):
    model.train(data=data, epochs=epoch, batch=batch, workers=workers, device=device, name=name)


def validate(model, data, batch, workers, device, name):
    model.val(data=data, batch=batch, workers=workers, device=device, name=name)


def main():
    model = YOLOv10(args.weights)
    if args.mode == 'train':
        train(model, args.data, args.epoch, args.batch, args.workers, args.device, args.name)
    else:
        validate(model, args.data, args.batch, args.workers, args.device, args.name)


if __name__ == '__main__':
    main()
