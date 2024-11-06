from ultralytics import YOLO

# 定义模型配置字典，方便选择不同的模型配置
MODEL_CONFIGS = {
    'yolov10n': r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n.yaml",
    'yolov10n_C2f_GhostModel_DynamicConv': r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n_C2f_GhostModel_DynamicConv.yaml",
    'yolov10n_SimAm': r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n-SimAm.yaml",  # SimAm模型
    'yolov10n_CBAM': r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n-CBAM.yaml",
    'yolov10n_BiFPN': r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n_BiFPN.yaml",
    'yolov10n_DynamicConv': r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n_DynamicConv.yaml",
    'yolov10n-CBMAWITHSimAm': r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10n-CBMAWITHSimAm.yaml",
    'yolov10m_BiFPN': r"D:\devProject\detect\yolov10\ultralytics\cfg\models\v10\yolov10m_BiFPN.yaml"
}


# 训练参数封装类
class YOLOTrainer:
    def __init__(self,
                 model_config_name: str,
                 data_yaml_path: str,
                 epochs: int = 300,
                 batch_size: int = 16,
                 img_size: int = 1024,
                 ):
        self.model_config_name = model_config_name
        self.data_yaml_path = data_yaml_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size

        # 从字典中选择模型配置路径
        self.model_yaml_path = self.get_model_yaml_path(model_config_name)
        if not self.model_yaml_path:
            raise ValueError(f"Model configuration '{model_config_name}' is not found!")

    def get_model_yaml_path(self, model_config_name: str):
        """根据配置名获取模型yaml文件路径"""
        return MODEL_CONFIGS.get(model_config_name)

    def train(self):
        """使用选择的模型配置进行训练"""
        print(f"Training using model config: {self.model_yaml_path}")
        model = YOLO(self.model_yaml_path)  # 加载模型
        results = model.train(data=self.data_yaml_path,
                              epochs=self.epochs,
                              batch=self.batch_size,
                              name=self.model_config_name,  # 自定义文件名
                              project="runs/train",  # 自定义文件保存路径
                              imgsz=self.img_size)
        return results


# 主程序
if __name__ == '__main__':
    # 选择要使用的模型配置
    selected_model_config = 'yolov10m_BiFPN'  # 这里可以选择模型，如 'yolov10n', 'yolov10n_SimAm', 等

    # 数据路径
    data_yaml_path = r'D:\devProject\detect\yolov10\datasets\GISDATA\data.yaml'

    # 创建训练器对象
    trainer = YOLOTrainer(model_config_name=selected_model_config, data_yaml_path=data_yaml_path)

    # 开始训练
    results = trainer.train()
