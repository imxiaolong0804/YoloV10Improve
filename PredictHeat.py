import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLOv10
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn


# --- 1. 辅助函数：自动查找多个候选层 ---
def find_target_layers(model, num_layers=3):
    """
    查找 YOLOv10 模型中多个候选目标层
    
    Args:
        model: YOLOv10 模型对象
        num_layers: 返回的候选层数量
        
    Returns:
        layers_info: [(layer, idx, layer_type), ...]
    """
    print("\n正在分析模型结构，查找候选卷积层...")
    
    # 获取模型的 model 属性（nn.Sequential）
    if hasattr(model, 'model'):
        model_layers = model.model
    else:
        model_layers = model
    
    candidate_layers = []
    
    # 从后往前遍历查找卷积层
    for idx in range(len(model_layers) - 1, -1, -1):
        layer = model_layers[idx]
        layer_type = type(layer).__name__
        
        # 跳过检测头
        if 'Detect' in layer_type:
            continue
        
        # 检查是否包含卷积操作
        has_conv = False
        if hasattr(layer, 'conv'):
            has_conv = True
        elif isinstance(layer, nn.Conv2d):
            has_conv = True
        else:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    has_conv = True
                    break
        
        if has_conv:
            candidate_layers.append((layer, idx, layer_type))
            print(f"  候选层 {len(candidate_layers)}: 索引={idx}, 类型={layer_type}")
            
            if len(candidate_layers) >= num_layers:
                break
    
    print(f"\n找到 {len(candidate_layers)} 个候选层")
    return candidate_layers


def find_last_conv_layer(model):
    """
    自动查找 YOLOv10 模型 Backbone 的最后一个卷积层
    
    Args:
        model: YOLOv10 模型对象
        
    Returns:
        target_layer: 找到的目标层
        layer_idx: 层索引
    """
    layers = find_target_layers(model, num_layers=1)
    if layers:
        return layers[0][0], layers[0][1]
    
    # 如果没找到，返回倒数第二层
    print("\n未找到明确的卷积层，使用倒数第二层作为默认值")
    model_layers = model.model if hasattr(model, 'model') else model
    return model_layers[-2], len(model_layers) - 2


# --- 2. 包装器：解决 YOLOv10 输出不是 Tensor 的问题 ---
class YOLOv10Wrapper(nn.Module):
    """
    包装 YOLOv10 模型以确保输出是 Tensor 格式
    """
    def __init__(self, model):
        super(YOLOv10Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 运行模型
        result = self.model(x)

        # 如果返回的是字典（YOLOv10 训练/推理模式）
        if isinstance(result, dict):
            # YOLOv10 可能包含 'one2one', 'one2many' 等键
            for key in ['one2many', 'one2one', 'output', 'preds']:
                if key in result:
                    val = result[key]
                    print(f"  使用字典键: {key}, 类型: {type(val)}")
                    
                    # 如果值是元组，取第一个元素（通常是预测 Tensor）
                    if isinstance(val, (tuple, list)):
                        print(f"  从元组/列表中提取第一个元素")
                        return val[0]
                    # 如果值已经是 Tensor，直接返回
                    elif isinstance(val, torch.Tensor):
                        return val
            
            # 如果找不到常见键，返回第一个是 Tensor 的值
            for key, val in result.items():
                if isinstance(val, torch.Tensor):
                    print(f"  使用第一个 Tensor 值: {key}")
                    return val
                elif isinstance(val, (tuple, list)):
                    for item in val:
                        if isinstance(item, torch.Tensor):
                            print(f"  从 {key} 的元组中提取 Tensor")
                            return item
        
        # YOLOv10 在推理模式下通常返回列表 [preds]
        if isinstance(result, (list, tuple)):
            print(f"  结果是元组/列表，长度: {len(result)}")
            return result[0]

        return result


# --- 3. 图像预处理 ---
def preprocess_image(img_path, imgsz=640):
    """
    预处理输入图像
    
    Args:
        img_path: 图片路径
        imgsz: 图片大小
        
    Returns:
        tensor: 处理后的张量
        rgb_img: RGB图像（归一化到0-1）
        original_img: 原始图像
    """
    # 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    
    print(f"原始图像大小: {img.shape}")
    
    # 调整大小
    img_resized = cv2.resize(img, (imgsz, imgsz))
    
    # 转换为 RGB 并归一化
    rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    rgb_img_normalized = np.float32(rgb_img) / 255.0
    
    # 转换为 PyTorch 张量 [1, 3, H, W]
    tensor = torch.from_numpy(rgb_img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    return tensor, rgb_img_normalized, img_resized


# --- 4. 生成热力图（增强版） ---
def generate_gradcam_heatmap(model_path, img_path, output_dir="runs/graduate/tests", imgsz=640, 
                            cam_method='eigencam', layer_idx=None, compare_methods=False):
    """
    使用 Grad-CAM 生成 YOLOv10 的热力图（增强版）
    
    Args:
        model_path: 模型权重路径
        img_path: 输入图片路径
        output_dir: 输出目录
        imgsz: 图像大小
        cam_method: CAM方法 ('gradcam', 'eigencam', 'gradcam++', 'xgradcam', 'layercam')
        layer_idx: 指定层索引，None则自动查找
        compare_methods: 是否生成多种方法对比图
        
    Returns:
        heatmap_path: 保存的热力图路径
    """
    print(f"\n{'='*60}")
    print(f"开始生成 YOLOv10 Grad-CAM 热力图")
    print(f"{'='*60}\n")
    
    # 1. 加载模型
    print(f"[1/5] 加载模型: {model_path}")
    yolo_model = YOLOv10(model_path)
    yolo_model.model.eval()  # 设置为评估模式
    
    device = next(yolo_model.model.parameters()).device
    print(f"  模型设备: {device}")
    
    # 2. 查找目标层
    print(f"\n[2/5] 查找目标层...")
    if layer_idx is not None:
        # 使用指定层
        model_layers = yolo_model.model.model if hasattr(yolo_model.model, 'model') else yolo_model.model
        target_layer = model_layers[layer_idx]
        print(f"  使用指定层索引: {layer_idx}")
    else:
        # 自动查找
        target_layer, layer_idx = find_last_conv_layer(yolo_model.model)
        print(f"  自动选择层索引: {layer_idx}")
    
    target_layers = [target_layer]
    
    # 3. 包装模型
    print(f"\n[3/5] 包装模型...")
    wrapped_model = YOLOv10Wrapper(yolo_model.model)
    
    # 4. 预处理图像
    print(f"\n[4/5] 预处理图像: {img_path}")
    input_tensor, rgb_img_normalized, img_bgr = preprocess_image(img_path, imgsz=imgsz)
    input_tensor = input_tensor.to(device)
    print(f"  输入张量形状: {input_tensor.shape}")
    
    # CAM方法映射
    cam_methods_map = {
        'gradcam': (GradCAM, "GradCAM"),
        'eigencam': (EigenCAM, "EigenCAM"),
        'gradcam++': (GradCAMPlusPlus, "GradCAM++"),
        'xgradcam': (XGradCAM, "XGradCAM"),
        'layercam': (LayerCAM, "LayerCAM")
    }
    
    # 5. 生成 Grad-CAM
    print(f"\n[5/5] 生成 Grad-CAM 热力图...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    img_name = Path(img_path).stem
    
    try:
        if compare_methods:
            # 生成多种方法对比
            print("  生成多种CAM方法对比...")
            heatmaps = []
            titles = []
            
            for method_key, (cam_class, method_name) in cam_methods_map.items():
                try:
                    print(f"    正在生成 {method_name}...")
                    cam = cam_class(model=wrapped_model, target_layers=target_layers)
                    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                    grayscale_cam = grayscale_cam[0, :]
                    heatmap = show_cam_on_image(rgb_img_normalized, grayscale_cam, use_rgb=True)
                    heatmaps.append(heatmap)
                    titles.append(method_name)
                    del cam
                except Exception as e:
                    print(f"    {method_name} 生成失败: {e}")
                    continue
            
            # 创建对比图
            if heatmaps:
                comparison = create_comparison_grid(rgb_img_normalized, heatmaps, titles)
                comparison_path = output_path / f"{img_name}_comparison.jpg"
                cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                print(f"\n✓ 对比图生成成功！")
                print(f"  保存路径: {comparison_path}")
                return str(comparison_path), comparison
        else:
            # 单一方法
            cam_class, method_name = cam_methods_map.get(cam_method.lower(), (EigenCAM, "EigenCAM"))
            print(f"  使用算法: {method_name}")
            
            cam = cam_class(model=wrapped_model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # 叠加热力图到原图
            heatmap_overlay = show_cam_on_image(rgb_img_normalized, grayscale_cam, use_rgb=True)
            
            # 保存结果
            heatmap_filename = f"{img_name}_{method_name.lower()}_layer{layer_idx}.jpg"
            heatmap_path = output_path / heatmap_filename
            
            # 保存热力图（转换回 BGR）
            heatmap_bgr = cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(heatmap_path), heatmap_bgr)
            
            # 同时保存纯热力图
            pure_heatmap = (grayscale_cam * 255).astype(np.uint8)
            pure_heatmap_colored = cv2.applyColorMap(pure_heatmap, cv2.COLORMAP_JET)
            pure_path = output_path / f"{img_name}_{method_name.lower()}_pure.jpg"
            cv2.imwrite(str(pure_path), pure_heatmap_colored)
            
            print(f"\n✓ 热力图生成成功！")
            print(f"  叠加图保存路径: {heatmap_path}")
            print(f"  纯热力图路径: {pure_path}")
            
            del cam
            return str(heatmap_path), heatmap_overlay
            
    except Exception as e:
        print(f"\n✗ 生成热力图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        torch.cuda.empty_cache()


def create_comparison_grid(original_img, heatmaps, titles):
    """
    创建多个热力图的对比网格
    
    Args:
        original_img: 原始图像（归一化）
        heatmaps: 热力图列表
        titles: 标题列表
        
    Returns:
        grid_img: 网格图像
    """
    n = len(heatmaps) + 1  # 包括原图
    cols = 3
    rows = (n + cols - 1) // cols
    
    h, w = heatmaps[0].shape[:2]
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    # 添加原图
    original_uint8 = (original_img * 255).astype(np.uint8)
    cv2.putText(original_uint8, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    grid[0:h, 0:w] = original_uint8
    
    # 添加热力图
    for idx, (heatmap, title) in enumerate(zip(heatmaps, titles), 1):
        row = idx // cols
        col = idx % cols
        heatmap_copy = heatmap.copy()
        cv2.putText(heatmap_copy, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = heatmap_copy
    
    return grid


# --- 5. 多层对比生成 ---
def compare_layers_heatmap(model_path, img_path, output_dir="runs/graduate/tests", imgsz=640, num_layers=3):
    """
    对比不同层的热力图效果
    
    Args:
        model_path: 模型权重路径
        img_path: 输入图片路径
        output_dir: 输出目录
        imgsz: 图像大小
        num_layers: 对比的层数
    """
    print(f"\n{'='*60}")
    print(f"生成多层热力图对比")
    print(f"{'='*60}\n")
    
    # 加载模型
    yolo_model = YOLOv10(model_path)
    yolo_model.model.eval()
    device = next(yolo_model.model.parameters()).device
    
    # 查找候选层
    candidate_layers = find_target_layers(yolo_model.model, num_layers=num_layers)
    
    if not candidate_layers:
        print("未找到候选层！")
        return
    
    # 预处理图像
    input_tensor, rgb_img_normalized, _ = preprocess_image(img_path, imgsz=imgsz)
    input_tensor = input_tensor.to(device)
    
    # 包装模型
    wrapped_model = YOLOv10Wrapper(yolo_model.model)
    
    # 生成每一层的热力图
    heatmaps = []
    titles = []
    
    for layer, idx, layer_type in candidate_layers:
        try:
            print(f"\n正在生成 Layer {idx} ({layer_type}) 的热力图...")
            cam = EigenCAM(model=wrapped_model, target_layers=[layer])
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            heatmap = show_cam_on_image(rgb_img_normalized, grayscale_cam, use_rgb=True)
            heatmaps.append(heatmap)
            titles.append(f"Layer{idx}-{layer_type}")
            del cam
        except Exception as e:
            print(f"  Layer {idx} 失败: {e}")
            continue
    
    # 创建对比图
    if heatmaps:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        img_name = Path(img_path).stem
        
        comparison = create_comparison_grid(rgb_img_normalized, heatmaps, titles)
        comparison_path = output_path / f"{img_name}_layers_comparison.jpg"
        cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        print(f"\n✓ 多层对比图生成成功！")
        print(f"  保存路径: {comparison_path}")
        return str(comparison_path)
    
    torch.cuda.empty_cache()


# --- 6. 批量处理 ---
def batch_generate_heatmaps(model_path, image_dir, output_dir="runs/graduate/tests", imgsz=640, cam_method='eigencam'):
    """
    批量生成热力图
    
    Args:
        model_path: 模型权重路径
        image_dir: 图片目录
        output_dir: 输出目录
        imgsz: 图像大小
        cam_method: CAM方法
    """
    image_dir = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 获取所有图片
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    print(f"\n找到 {len(image_files)} 张图片")
    
    # 批量处理
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n处理 [{idx}/{len(image_files)}]: {img_path.name}")
        try:
            generate_gradcam_heatmap(model_path, img_path, output_dir, imgsz, cam_method=cam_method)
        except Exception as e:
            print(f"  跳过此图片，错误: {e}")
            continue
    
    print(f"\n全部完成！共处理 {len(image_files)} 张图片")


if __name__ == "__main__":
    # --- 配置区域 ---
    weights_path = r"D:\devProject\detect\yolov10\runs\graduate\train\yolov10n\weights\best.pt"
    image_source = r"D:\devProject\detect\yolov10\datasets\data\test\images\2_image349_7f2d7baf.png"
    output_directory = "runs/graduate/tests"
    
    # ========== 选项 1: 单张图片 - 单一方法 ==========
    print("\n" + "="*60)
    print("模式 1: 单张图片 - 单一方法")
    print("="*60)
    try:
        heatmap_path, _ = generate_gradcam_heatmap(
            model_path=weights_path,
            img_path=image_source,
            output_dir=output_directory,
            imgsz=640,
            cam_method='eigencam',  # 可选: 'gradcam', 'eigencam', 'gradcam++', 'xgradcam', 'layercam'
            layer_idx=None,  # None=自动查找，或指定层索引如 22
            compare_methods=False
        )
    except Exception as e:
        print(f"\n程序异常: {e}")
    
    # ========== 选项 2: 单张图片 - 多方法对比 ==========
    print("\n" + "="*60)
    print("模式 2: 单张图片 - 多种CAM方法对比")
    print("="*60)
    try:
        comparison_path, _ = generate_gradcam_heatmap(
            model_path=weights_path,
            img_path=image_source,
            output_dir=output_directory,
            imgsz=640,
            compare_methods=True  # 启用多方法对比
        )
    except Exception as e:
        print(f"\n程序异常: {e}")
    
    # ========== 选项 3: 单张图片 - 多层对比 ==========
    print("\n" + "="*60)
    print("模式 3: 单张图片 - 多层对比")
    print("="*60)
    try:
        layers_comparison_path = compare_layers_heatmap(
            model_path=weights_path,
            img_path=image_source,
            output_dir=output_directory,
            imgsz=640,
            num_layers=3  # 对比的层数
        )
    except Exception as e:
        print(f"\n程序异常: {e}")
    
    # ========== 选项 4: 批量处理（取消注释以启用） ==========
    # batch_generate_heatmaps(
    #     model_path=weights_path,
    #     image_dir=r"D:\devProject\detect\yolov10\datasets\data\test\images",
    #     output_dir=output_directory,
    #     imgsz=640,
    #     cam_method='eigencam'
    # )
    
    print("\n" + "="*60)
    print("所有任务完成！")
    print(f"请查看输出目录: {output_directory}")
    print("="*60)
