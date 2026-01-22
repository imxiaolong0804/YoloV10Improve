from ultralytics import YOLOv10, YOLO
import os
from pathlib import Path
from datetime import datetime


class YOLOPredictor:
    """YOLO批量预测器"""
    
    def __init__(self,
                 source: str,
                 models_dir: str = "runs/bxl/models",
                 save_path: str = "runs/bxl/predict",
                 img_size: int = 1024,
                 conf: float = 0.25,
                 save: bool = True,
                 visualize: bool = False):
        """
        初始化预测器
        
        Args:
            source: 要检测的图片/视频/文件夹路径
            models_dir: 模型权重所在的基础目录（包含多个模型子目录）
            save_path: 预测结果保存路径
            img_size: 输入图像大小
            conf: 置信度阈值
            save: 是否保存预测结果
            visualize: 是否可视化特征图
        """
        self.source = source
        self.models_dir = models_dir
        self.save_path = save_path
        self.img_size = img_size
        self.conf = conf
        self.save = save
        self.visualize = visualize
    
    def find_model_weights(self):
        """
        在models_dir目录下查找所有模型的best.pt权重文件
        
        Returns:
            dict: {模型名称: 权重文件路径}
        """
        model_weights = {}
        models_path = Path(self.models_dir)
        
        if not models_path.exists():
            print(f"警告: 模型目录不存在 - {self.models_dir}")
            return model_weights
        
        # 遍历模型目录下的所有子文件夹
        for model_dir in models_path.iterdir():
            if model_dir.is_dir():
                # 查找 weights/best.pt
                best_weight = model_dir / "weights" / "best.pt"
                if best_weight.exists():
                    model_weights[model_dir.name] = str(best_weight)
                else:
                    # 尝试查找 weights/last.pt
                    last_weight = model_dir / "weights" / "last.pt"
                    if last_weight.exists():
                        model_weights[model_dir.name] = str(last_weight)
        
        return model_weights
    
    def predict_single(self, model_name: str, weight_path: str):
        """
        使用单个模型进行预测
        
        Args:
            model_name: 模型名称（用于保存结果）
            weight_path: 模型权重路径
        """
        print(f"\n{'='*60}")
        print(f"正在使用模型进行预测: {model_name}")
        print(f"权重文件: {weight_path}")
        print(f"{'='*60}\n")
        
        try:
            # 加载模型（使用YOLOv10类以兼容v10模型的特殊输出格式）
            model = YOLOv10(weight_path)
            
            # 执行预测
            results = model.predict(
                source=self.source,
                save=self.save,
                imgsz=self.img_size,
                conf=self.conf,
                project=self.save_path,
                name=model_name,
                visualize=self.visualize
            )
            
            print(f"\n✓ 模型 {model_name} 预测完成！")
            print(f"  结果保存在: {self.save_path}/{model_name}")
            
            return {
                'model': model_name,
                'status': 'success',
                'results_count': len(results) if results else 0
            }
            
        except Exception as e:
            print(f"\n✗ 模型 {model_name} 预测失败！")
            print(f"  错误信息: {e}")
            
            return {
                'model': model_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def predict_all(self, model_names: list = None):
        """
        使用所有找到的模型进行批量预测
        
        Args:
            model_names: 指定要使用的模型名称列表，None则使用所有找到的模型
        
        Returns:
            list: 所有预测结果的汇总
        """
        # 查找所有模型权重
        all_weights = self.find_model_weights()
        
        if not all_weights:
            print(f"错误: 在 {self.models_dir} 目录下没有找到任何模型权重！")
            return []
        
        # 筛选要使用的模型
        if model_names:
            weights_to_use = {k: v for k, v in all_weights.items() if k in model_names}
        else:
            weights_to_use = all_weights
        
        print(f"\n{'#'*80}")
        print(f"# 批量预测任务开始")
        print(f"# 数据源: {self.source}")
        print(f"# 共找到 {len(weights_to_use)} 个模型")
        print(f"# 模型列表: {', '.join(weights_to_use.keys())}")
        print(f"{'#'*80}\n")
        
        # 记录所有预测结果
        all_results = []
        
        # 轮询每个模型进行预测
        for idx, (model_name, weight_path) in enumerate(weights_to_use.items(), 1):
            print(f"\n\n{'#'*80}")
            print(f"# [{idx}/{len(weights_to_use)}] 预测任务: {model_name}")
            print(f"{'#'*80}\n")
            
            result = self.predict_single(model_name, weight_path)
            all_results.append(result)
            
            # 清理显存
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # 打印最终汇总
        print(f"\n\n{'#'*80}")
        print(f"# 批量预测任务完成")
        print(f"{'#'*80}")
        print(f"\n预测结果汇总:")
        print("-" * 80)
        
        success_count = sum(1 for r in all_results if r['status'] == 'success')
        failed_count = len(all_results) - success_count
        
        print(f"总计: {len(all_results)} 个模型")
        print(f"成功: {success_count} 个")
        print(f"失败: {failed_count} 个\n")
        
        for result in all_results:
            status_icon = "✓" if result['status'] == 'success' else "✗"
            print(f"{status_icon} {result['model']}: {result['status']}")
            if result['status'] == 'success':
                print(f"  └─ 检测图像数: {result.get('results_count', 'N/A')}")
        
        print(f"\n结果保存目录: {self.save_path}")
        print(f"{'#'*80}\n")
        
        return all_results


def batch_predict(source: str,
                  models_dir: str = "runs/bxl/models",
                  save_path: str = "runs/bxl/predict",
                  img_size: int = 1024,
                  conf: float = 0.25,
                  model_names: list = None):
    """
    批量预测便捷函数
    
    Args:
        source: 要检测的图片/视频/文件夹路径
        models_dir: 模型权重所在的基础目录
        save_path: 预测结果保存路径
        img_size: 输入图像大小
        conf: 置信度阈值
        model_names: 指定要使用的模型名称列表，None则使用所有找到的模型
    """
    predictor = YOLOPredictor(
        source=source,
        models_dir=models_dir,
        save_path=save_path,
        img_size=img_size,
        conf=conf
    )
    
    return predictor.predict_all(model_names)


# ==================== 主程序 ====================
if __name__ == '__main__':
    # ========== 配置区域 ==========
    
    # 要检测的数据源（可以是图片、视频或文件夹）
    SOURCE = r'D:\devProject\detect\yolov10\datasets\data\test\images'
    
    # 模型权重目录（训练保存的模型）
    MODELS_DIR = "runs/bxl/models"
    
    # 预测结果保存路径
    SAVE_PATH = "runs/bxl/predict"
    
    # 预测参数
    IMG_SIZE = 1024
    CONF = 0.25  # 置信度阈值
    
    # ========== 选择预测模式 ==========
    print("\n选择预测模式:")
    print("1. 使用所有模型进行预测")
    print("2. 选择指定模型进行预测")
    print("3. 单个模型预测（手动指定权重路径）")
    
    mode = input("\n请输入选项 (1/2/3，默认1): ").strip() or "1"
    
    if mode == "1":
        # 使用所有模型进行批量预测
        print("\n将使用所有找到的模型进行预测...\n")
        batch_predict(
            source=SOURCE,
            models_dir=MODELS_DIR,
            save_path=SAVE_PATH,
            img_size=IMG_SIZE,
            conf=CONF
        )
    
    elif mode == "2":
        # 选择指定模型
        predictor = YOLOPredictor(
            source=SOURCE,
            models_dir=MODELS_DIR,
            save_path=SAVE_PATH,
            img_size=IMG_SIZE,
            conf=CONF
        )
        
        # 查找所有可用模型
        available_models = predictor.find_model_weights()
        
        if not available_models:
            print(f"\n错误: 在 {MODELS_DIR} 目录下没有找到任何模型！")
        else:
            print("\n可用的模型:")
            model_list = list(available_models.keys())
            for idx, model_name in enumerate(model_list, 1):
                print(f"{idx}. {model_name}")
            
            print("\n请输入要使用的模型编号，用逗号分隔（例如: 1,3,5）")
            selected = input("模型编号: ").strip()
            
            if selected:
                try:
                    indices = [int(i.strip()) - 1 for i in selected.split(',')]
                    selected_models = [model_list[i] for i in indices if 0 <= i < len(model_list)]
                    
                    if selected_models:
                        print(f"\n将使用以下模型: {', '.join(selected_models)}\n")
                        predictor.predict_all(model_names=selected_models)
                    else:
                        print("错误: 没有有效的模型选择！")
                except (ValueError, IndexError) as e:
                    print(f"错误: 输入格式不正确 - {e}")
            else:
                print("未选择任何模型，退出。")
    
    elif mode == "3":
        # 单个模型预测（手动指定权重路径）
        weight_path = input("\n请输入模型权重路径: ").strip()
        
        if weight_path and os.path.exists(weight_path):
            model = YOLOv10(weight_path)
            
            model.predict(
                source=SOURCE,
                save=True,
                imgsz=IMG_SIZE,
                conf=CONF,
                project=SAVE_PATH,
                name="single_predict"
            )
            
            print(f"\n预测完成！结果保存在: {SAVE_PATH}/single_predict")
        else:
            print("错误: 权重文件路径无效！")
    
    else:
        print("无效的选项，退出。")