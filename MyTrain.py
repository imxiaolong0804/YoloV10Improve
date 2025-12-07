import warnings

from ultralytics import YOLO
import os


warnings.filterwarnings('ignore')

# 获取当前文件所在目录
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultralytics", "cfg", "models", "v10")

MODEL_CONFIGS = {
    'yolov10n': os.path.join(BASE_DIR, "yolov10n.yaml"),
    'yolov10n_C2f_GhostModel_DynamicConv': os.path.join(BASE_DIR, "yolov10n_C2f_GhostModel_DynamicConv.yaml"),
    'yolov10n_SimAm': os.path.join(BASE_DIR, "yolov10n-SimAm.yaml"),
    'yolov10n_CBAM': os.path.join(BASE_DIR, "yolov10n-CBAM.yaml"),
    'yolov10n_BiFPN': os.path.join(BASE_DIR, "yolov10n_BiFPN.yaml"),
    'yolov10n_DynamicConv': os.path.join(BASE_DIR, "yolov10n_DynamicConv.yaml"),
    'yolov10n-CBMAWITHSimAm': os.path.join(BASE_DIR, "yolov10n-CBMAWITHSimAm.yaml"),
    'yolov10m_BiFPN': os.path.join(BASE_DIR, "yolov10m_BiFPN.yaml"),
    'yolov10l': os.path.join(BASE_DIR, "yolov10l.yaml")
}


# 训练参数封装类
def get_model_yaml_path(model_config_name: str):
    """根据配置名获取模型yaml文件路径"""
    return MODEL_CONFIGS.get(model_config_name)


class YOLOTrainer:
    def __init__(self,
                 model_config_name: str,
                 data_yaml_path: str,
                 save_path: str,
                 epochs: int = 200,
                 batch_size: int = 32,
                 img_size: int = 1024,
                 weights_path: str = None  # 允许指定继续训练的模型权重文件路径
                 ):
        self.model_config_name = model_config_name
        self.data_yaml_path = data_yaml_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.save_path = save_path
        self.weights_path = weights_path  # 新增：用于传入继续训练的权重路径

        # 从字典中选择模型配置路径
        self.model_yaml_path = get_model_yaml_path(model_config_name)
        if not self.model_yaml_path:
            raise ValueError(f"Model configuration '{model_config_name}' is not found!")

    def train(self):
        """使用选择的模型配置进行训练"""
        print(f"Training using model config: {self.model_yaml_path}")

        # 加载YOLO模型，传入继续训练的权重文件（如果有）
        model = YOLO(self.model_yaml_path)

        # 如果指定了继续训练的权重路径，传入给模型
        if self.weights_path:
            print(f"Resuming training from checkpoint: {self.weights_path}")
            model.load(self.weights_path)

        results = model.train(
            data=self.data_yaml_path,
            epochs=self.epochs,
            batch=self.batch_size,
            name=self.model_config_name,  # 自定义文件名
            project=self.save_path,  # 自定义文件保存路径
            imgsz=self.img_size,
        )
        return results


# 主程序
if __name__ == '__main__':
    # 选择要使用的模型配置
    selected_model_config = 'yolov10n'  # 这里可以选择模型，如 'yolov10n', 'yolov10n_SimAm', 等

    # 数据路径
    data_yaml_path = r'D:/devProject/detect/yolov10/datasets/data/data.yaml'

    # 如果训练已经中断并且有之前的权重，可以传入 `weights_path`
    # last_checkpoint_path = r"./yolov10n.pt"  # 或者 best.pt

    # 创建训练器对象，并指定继续训练的权重
    trainer = YOLOTrainer(
        model_config_name=selected_model_config,
        data_yaml_path=data_yaml_path,
        # weights_path=last_checkpoint_path,  # 指定继续训练的权重路径
        save_path="runs/graduate/train",
        epochs=300
    )

    # 开始训练
    results = trainer.train()

# 导出
# yolo export model=D:/devProject/detect/yolov10/runs/train/yolov10n/weights/best.pt format=onnx

# 导出
# yolo export model=D:/devProject/detect/yolov10/runs/train/yolov10m_BiFPN3/weights/best.pt format=onnx
