from ultralytics import YOLOv10

model = YOLOv10(r"D:\devProject\detect\yolov10\runs\detect\raw_model\weights\best.pt")

model.predict(
    source=r"D:\devProject\detect\yolov10\ultralytics\assets\4_jpg.rf.cb877fa409bbf186684996b959cb485b_aug_1.jpg",
    save=True,
    imgsz=1024,
    project='runs/predict'
)