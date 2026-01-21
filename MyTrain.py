import warnings
from ultralytics import YOLO
import os
import json
import csv
from datetime import datetime
from pathlib import Path

# warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)

# 获取当前文件所在目录
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultralytics", "cfg", "models", "v10")

MODEL_CONFIGS = {
    'yolov10n': os.path.join(BASE_DIR, "yolov10n.yaml"),
    'yolov10n_MCAttn': os.path.join(BASE_DIR, "yolov10n-MCAttn.yaml"),
    'yolov10n_PPA': os.path.join(BASE_DIR, "yolov10n-PPA.yaml"),
    'yolov10n_CSP': os.path.join(BASE_DIR, "yolov10n-CSPStage.yaml"),
    'yolov10n-MCAttnPPA': os.path.join(BASE_DIR, "yolov10n-MCAttnPPA.yaml"),
    'yolov10n-MCAttn-CSP': os.path.join(BASE_DIR, "yolov10n-MCAttn-CSP.yaml"),
    'yolov10n-MCAttn-PPA-CSPStage.yaml': os.path.join(BASE_DIR, "yolov10n-MCAttn-PPA-CSPStage.yaml"),
    'yolov10n_SimAm': os.path.join(BASE_DIR, "yolov10n-SimAm.yaml"),
    'yolov10n_C2f_GhostModel_DynamicConv': os.path.join(BASE_DIR, "yolov10n_C2f_GhostModel_DynamicConv.yaml"),
    'yolov10n_CBAM': os.path.join(BASE_DIR, "yolov10n-CBAM.yaml"),
    'yolov10n_BiFPN': os.path.join(BASE_DIR, "yolov10n_BiFPN.yaml"),
    'yolov10n_DynamicConv': os.path.join(BASE_DIR, "yolov10n_DynamicConv.yaml"),
    'yolov10n-CBMAWITHSimAm': os.path.join(BASE_DIR, "yolov10n-CBMAWITHSimAm.yaml"),
    'yolov10m_BiFPN': os.path.join(BASE_DIR, "yolov10m_BiFPN.yaml"),
    # 'yolov10l': os.path.join(BASE_DIR, "yolov10l.yaml")
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
                 weights_path: str = None,  # 允许指定继续训练的模型权重文件路径
                 log_dir: str = "runs/training_logs"  # 日志保存目录
                 ):
        self.model_config_name = model_config_name
        self.data_yaml_path = data_yaml_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.save_path = save_path
        self.weights_path = weights_path  # 新增：用于传入继续训练的权重路径
        self.log_dir = log_dir  # 日志目录

        # 从字典中选择模型配置路径
        self.model_yaml_path = get_model_yaml_path(model_config_name)
        if not self.model_yaml_path:
            raise ValueError(f"Model configuration '{model_config_name}' is not found!")
        
        # 创建日志目录
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def train(self):
        """使用选择的模型配置进行训练"""
        print(f"\n{'='*80}")
        print(f"开始训练模型: {self.model_config_name}")
        print(f"配置文件: {self.model_yaml_path}")
        print(f"{'='*80}\n")

        # 1. 始终使用您选择的 .yaml 配置来构建模型结构
        model = YOLO(self.model_yaml_path)

        # 2. 确定初始权重文件路径
        if self.weights_path:
            # 如果指定了继续训练的权重路径，使用它
            initial_weights = self.weights_path
            print(f"从检查点继续训练: {initial_weights}")
        else:
            # 否则，使用对应的官方预训练权重 (例如：'yolov10n.pt')
            initial_weights = "yolov10n.pt"
            print(f"使用官方预训练权重: {initial_weights}")

        # 3. 将权重文件加载到已经构建好的自定义结构模型中
        model.load(initial_weights)

        # 4. 开始训练
        print(f"\n开始训练，训练轮数: {self.epochs}\n")
        results = model.train(
            data=self.data_yaml_path,
            epochs=self.epochs,
            batch=self.batch_size,
            name=self.model_config_name,
            project=self.save_path,
            imgsz=self.img_size,
        )
        
        # 5. 训练完成后进行验证并获取详细指标
        print(f"\n{'='*80}")
        print(f"训练完成，开始验证模型: {self.model_config_name}")
        print(f"{'='*80}\n")
        
        # 加载最佳权重进行验证
        best_model_path = os.path.join(self.save_path, self.model_config_name, "weights", "best.pt")
        best_model = YOLO(best_model_path)
        
        # 验证模型
        val_results = best_model.val(data=self.data_yaml_path)
        
        # 6. 收集并保存详细指标
        training_info = self._collect_training_info(best_model, val_results, best_model_path)
        self._save_training_info(training_info)
        
        print(f"\n{'='*80}")
        print(f"模型 {self.model_config_name} 训练完成！")
        print(f"最佳权重: {best_model_path}")
        print(f"{'='*80}\n")
        
        return results, training_info
    
    def _collect_training_info(self, model, val_results, model_path):
        """收集训练信息和模型指标"""
        training_info = {
            "model_name": self.model_config_name,
            "model_yaml": self.model_yaml_path,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "img_size": self.img_size,
            "data_yaml": self.data_yaml_path,
            "best_model_path": model_path,
        }
        
        # 获取验证结果
        try:
            # 总体指标
            training_info["metrics"] = {
                "mAP50": float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0.0,
                "mAP50-95": float(val_results.box.map) if hasattr(val_results.box, 'map') else 0.0,
                "precision": float(val_results.box.mp) if hasattr(val_results.box, 'mp') else 0.0,
                "recall": float(val_results.box.mr) if hasattr(val_results.box, 'mr') else 0.0,
            }
            
            # 每个类别的指标
            if hasattr(val_results.box, 'maps'):
                class_names = model.names if hasattr(model, 'names') else {}
                per_class_metrics = {}
                
                for idx, (ap50, ap) in enumerate(zip(val_results.box.ap50, val_results.box.ap)):
                    class_name = class_names.get(idx, f"class_{idx}")
                    per_class_metrics[class_name] = {
                        "mAP50": float(ap50) if ap50 is not None else 0.0,
                        "mAP50-95": float(ap) if ap is not None else 0.0,
                    }
                
                training_info["per_class_metrics"] = per_class_metrics
            
            print("\n" + "="*60)
            print("验证指标汇总:")
            print("="*60)
            print(f"mAP@0.5: {training_info['metrics']['mAP50']:.4f}")
            print(f"mAP@0.5:0.95: {training_info['metrics']['mAP50-95']:.4f}")
            print(f"Precision: {training_info['metrics']['precision']:.4f}")
            print(f"Recall: {training_info['metrics']['recall']:.4f}")
            
            if "per_class_metrics" in training_info:
                print("\n每类指标:")
                print("-" * 60)
                for class_name, metrics in training_info["per_class_metrics"].items():
                    print(f"  {class_name}:")
                    print(f"    mAP@0.5: {metrics['mAP50']:.4f}")
                    print(f"    mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            
        except Exception as e:
            print(f"警告: 收集验证指标时出错: {e}")
            training_info["metrics"] = {"error": str(e)}
        
        # 获取模型大小和FLOPs
        try:
            # 获取模型文件大小
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            training_info["model_size_mb"] = round(model_size_mb, 2)
            
            # 获取模型信息（包括参数量和FLOPs）
            model_info = model.info(verbose=False)
            training_info["parameters"] = model_info[1] if isinstance(model_info, tuple) else "N/A"
            training_info["gflops"] = model_info[2] if isinstance(model_info, tuple) and len(model_info) > 2 else "N/A"
            
            print("\n" + "="*60)
            print("模型信息:")
            print("="*60)
            print(f"模型大小: {training_info['model_size_mb']} MB")
            print(f"参数量: {training_info['parameters']}")
            print(f"GFLOPs: {training_info['gflops']}")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"警告: 获取模型信息时出错: {e}")
            training_info["model_size_mb"] = "N/A"
            training_info["parameters"] = "N/A"
            training_info["gflops"] = "N/A"
        
        return training_info
    
    def _save_training_info(self, training_info):
        """保存训练信息到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存为JSON格式（详细信息）
        json_filename = f"{self.model_config_name}_{timestamp}.json"
        json_path = os.path.join(self.log_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=4, ensure_ascii=False)
        
        print(f"详细训练信息已保存到: {json_path}")
        
        # 2. 追加到汇总CSV文件
        csv_path = os.path.join(self.log_dir, "training_summary.csv")
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = [
                'model_name', 'training_date', 'epochs', 'batch_size', 'img_size',
                'mAP50', 'mAP50-95', 'precision', 'recall',
                'model_size_mb', 'parameters', 'gflops', 'best_model_path'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            row = {
                'model_name': training_info['model_name'],
                'training_date': training_info['training_date'],
                'epochs': training_info['epochs'],
                'batch_size': training_info['batch_size'],
                'img_size': training_info['img_size'],
                'mAP50': training_info['metrics'].get('mAP50', 'N/A'),
                'mAP50-95': training_info['metrics'].get('mAP50-95', 'N/A'),
                'precision': training_info['metrics'].get('precision', 'N/A'),
                'recall': training_info['metrics'].get('recall', 'N/A'),
                'model_size_mb': training_info.get('model_size_mb', 'N/A'),
                'parameters': training_info.get('parameters', 'N/A'),
                'gflops': training_info.get('gflops', 'N/A'),
                'best_model_path': training_info['best_model_path']
            }
            writer.writerow(row)
        
        print(f"训练摘要已追加到: {csv_path}")
        
        # 3. 如果有每类指标，保存到单独的CSV
        if 'per_class_metrics' in training_info:
            per_class_csv = os.path.join(self.log_dir, f"{self.model_config_name}_{timestamp}_per_class.csv")
            with open(per_class_csv, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['class_name', 'mAP50', 'mAP50-95']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for class_name, metrics in training_info['per_class_metrics'].items():
                    writer.writerow({
                        'class_name': class_name,
                        'mAP50': metrics['mAP50'],
                        'mAP50-95': metrics['mAP50-95']
                    })
            
            print(f"每类指标已保存到: {per_class_csv}")


# ==================== 批量训练函数 ====================
def batch_train_models(model_configs_to_train=None, 
                       data_yaml_path=None,
                       save_path="runs/bxl/train",
                       log_dir="runs/training_logs",
                       epochs=180,
                       batch_size=32,
                       img_size=1024):
    """
    批量训练多个模型配置
    
    Args:
        model_configs_to_train: 要训练的模型配置列表，None则训练所有
        data_yaml_path: 数据配置文件路径
        save_path: 训练结果保存基础路径
        log_dir: 日志保存目录
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像大小
    """
    # 如果没有指定要训练的模型，则训练所有模型
    if model_configs_to_train is None:
        model_configs_to_train = list(MODEL_CONFIGS.keys())
    
    # 如果没有指定数据路径，使用默认路径
    if data_yaml_path is None:
        data_yaml_path = r'D:\devProject\detect\yolov10\datasets\data\data.yaml'
    
    print(f"\n{'#'*80}")
    print(f"# 批量训练任务开始")
    print(f"# 总共需要训练 {len(model_configs_to_train)} 个模型")
    print(f"# 模型列表: {', '.join(model_configs_to_train)}")
    print(f"{'#'*80}\n")
    
    # 记录所有训练结果
    all_results = []
    
    # 循环训练每个模型
    for idx, model_config_name in enumerate(model_configs_to_train, 1):
        print(f"\n\n{'#'*80}")
        print(f"# [{idx}/{len(model_configs_to_train)}] 开始训练: {model_config_name}")
        print(f"{'#'*80}\n")
        
        try:
            # 为每个模型创建独立的保存路径
            model_save_path = f"{save_path}_{model_config_name}"
            
            # 创建训练器
            trainer = YOLOTrainer(
                model_config_name=model_config_name,
                data_yaml_path=data_yaml_path,
                save_path=model_save_path,
                log_dir=log_dir,
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size
            )
            
            # 开始训练
            results, training_info = trainer.train()
            
            # 记录结果
            all_results.append({
                'model': model_config_name,
                'status': 'success',
                'info': training_info
            })
            
            print(f"\n{'='*80}")
            print(f"✓ 模型 {model_config_name} 训练成功！")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"✗ 模型 {model_config_name} 训练失败！")
            print(f"错误信息: {e}")
            print(f"{'='*80}\n")
            
            all_results.append({
                'model': model_config_name,
                'status': 'failed',
                'error': str(e)
            })
            
            import traceback
            traceback.print_exc()
            
            # 继续训练下一个模型
            continue
    
    # 打印最终汇总
    print(f"\n\n{'#'*80}")
    print(f"# 批量训练任务完成")
    print(f"{'#'*80}")
    print(f"\n训练结果汇总:")
    print("-" * 80)
    
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    failed_count = len(all_results) - success_count
    
    print(f"总计: {len(all_results)} 个模型")
    print(f"成功: {success_count} 个")
    print(f"失败: {failed_count} 个\n")
    
    for result in all_results:
        status_icon = "✓" if result['status'] == 'success' else "✗"
        print(f"{status_icon} {result['model']}: {result['status']}")
        if result['status'] == 'success' and 'info' in result:
            metrics = result['info'].get('metrics', {})
            print(f"  └─ mAP@0.5: {metrics.get('mAP50', 'N/A'):.4f}, "
                  f"mAP@0.5:0.95: {metrics.get('mAP50-95', 'N/A'):.4f}")
    
    print(f"\n日志目录: {log_dir}")
    print(f"训练结果目录: {save_path}_*")
    print(f"{'#'*80}\n")
    
    return all_results


# ==================== 主程序 ====================
if __name__ == '__main__':
    # ========== 配置区域 ==========
    
    # 数据路径
    data_yaml_path = r'D:\devProject\detect\yolov10\datasets\data\data.yaml'
    # data_yaml_path = r'D:\devProject\detect\yolov10\datasets\GISDATA\data.yaml'
    
    # 训练参数
    EPOCHS = 180
    BATCH_SIZE = 32
    IMG_SIZE = 1024
    SAVE_PATH = "runs/bxl/train"
    LOG_DIR = "runs/training_logs"
    
    # ========== 选项 1: 训练所有模型 ==========
    print("\n选择训练模式:")
    print("1. 训练所有模型")
    print("2. 训练指定模型")
    print("3. 训练单个模型（原有方式）")
    
    mode = input("\n请输入选项 (1/2/3，默认1): ").strip() or "1"
    
    if mode == "1":
        # 训练所有模型
        print("\n将训练所有模型配置...\n")
        batch_train_models(
            model_configs_to_train=None,  # None表示训练所有
            data_yaml_path=data_yaml_path,
            save_path=SAVE_PATH,
            log_dir=LOG_DIR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE
        )
    
    elif mode == "2":
        # 训练指定的模型列表
        print("\n可用的模型配置:")
        for idx, model_name in enumerate(MODEL_CONFIGS.keys(), 1):
            print(f"{idx}. {model_name}")
        
        print("\n请输入要训练的模型编号，用逗号分隔（例如: 1,3,5）")
        selected = input("模型编号: ").strip()
        
        if selected:
            try:
                indices = [int(i.strip()) - 1 for i in selected.split(',')]
                model_list = list(MODEL_CONFIGS.keys())
                selected_models = [model_list[i] for i in indices if 0 <= i < len(model_list)]
                
                if selected_models:
                    print(f"\n将训练以下模型: {', '.join(selected_models)}\n")
                    batch_train_models(
                        model_configs_to_train=selected_models,
                        data_yaml_path=data_yaml_path,
                        save_path=SAVE_PATH,
                        log_dir=LOG_DIR,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        img_size=IMG_SIZE
                    )
                else:
                    print("错误: 没有有效的模型选择！")
            except (ValueError, IndexError) as e:
                print(f"错误: 输入格式不正确 - {e}")
        else:
            print("未选择任何模型，退出。")
    
    elif mode == "3":
        # 单个模型训练（原有方式）
        selected_model_config = 'yolov10n'
        
        print(f"\n训练单个模型: {selected_model_config}\n")
        
        trainer = YOLOTrainer(
            model_config_name=selected_model_config,
            data_yaml_path=data_yaml_path,
            save_path=f"{SAVE_PATH}_{selected_model_config}",
            log_dir=LOG_DIR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE
        )
        
        results, training_info = trainer.train()
    
    else:
        print("无效的选项，退出。")

# 导出
# yolo export model=D:/devProject/detect/yolov10/runs/train/yolov10n/weights/best.pt format=onnx
