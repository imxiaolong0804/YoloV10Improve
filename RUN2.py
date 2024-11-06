from ultralytics import YOLOv10

# model_yaml_path = r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10_BiFPN.yaml"
# model_yaml_path = r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n.yaml"
# model_yaml_path = r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n-CBMAWITHSimAm.yaml"
# model_yaml_path = r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n_DynamicConv.yaml"
model_yaml_path = r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n_C2f_GhostModel_DynamicConv.yaml"
# SimAm
# model_yaml_path = r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n-SimAm.yaml" // 暂时好像没有修复
data_yaml_path = r'D:\devProject\detect\yolov10\datasets\GISDATA\data.yaml'

if __name__ == '__main__':
    model = YOLOv10(model_yaml_path)
    results = model.train(data=data_yaml_path,
                          epochs=10,
                          batch=32,
                          name='Dynamic_model',  # 自定义文件名
                          project="runs/train",  # 自定义文件保存路径
                          # freeze=10,  # 冻结前n层
                          imgsz=1024)
