from ultralytics import YOLOv10, YOLO

model = YOLOv10(r"D:\devProject\detect\yolov10\runs\graduate\train\yolov10n\weights\best.pt")

model.predict(
    # source=r"D:\devProject\detect\yolov10\ultralytics\assets\4_jpg.rf.cb877fa409bbf186684996b959cb485b_aug_1.jpg",
    # source=r"D:\devProject\detect\yolov10\datasets\try",
    source=r"D:\devProject\detect\yolov10\datasets\data\test\images\2_image349_7f2d7baf.png",
    save=True,
    imgsz=1024,
    project='runs/graduate/predict/tests',
    visualize=True
)