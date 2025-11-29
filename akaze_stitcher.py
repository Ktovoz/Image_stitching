"""
垂直长图拼接工具 - 基于AKAZE特征点的智能图像拼接

Author: Ktovoz
Date: 2025-11-28

基本原理：
1. AKAZE特征提取: 使用OpenCV的AKAZE算法提取每张图像的关键点和描述符
2. 特征匹配: 使用BFMatcher匹配器对相邻图像的特征点进行匹配
3. 几何验证: 应用单应性矩阵验证，确保正确的空间关系
4. 区域分析: 分析匹配点的分布，确定相对位置
5. 智能裁剪: 基于匹配信息自动裁剪重叠区域
6. 垂直拼接: 将裁剪后的图像垂直堆叠，创建最终全景图

特点：
- 自动检测图像顺序（文件名排序或自定义顺序）
- 基于特征点的智能拼接，避免简单的垂直堆叠
- 生成特征匹配可视化，帮助调试拼接质量
- 支持多种图像格式（jpg, jpeg, png, bmp）
- 详细的进度反馈和错误处理
- 智能裁剪分析，最小化输出文件大小

使用场景：
- 手机截图拼接成长图
- 文档扫描图片的垂直拼接
- 网页长截图的自动化拼接
- 从多个图像段创建全景图
"""

import os
import sys
import json
from typing import Optional, Dict
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from stitcher_module.stitcher import StandaloneImageStitcher


def load_custom_order(order_file: str) -> Optional[Dict[str, int]]:
    if not os.path.exists(order_file):
        logger.error(f"order file '{order_file}' does not exist")
        return None

    try:
        with open(order_file, 'r', encoding='utf-8') as f:
            custom_order = json.load(f)
        
        return custom_order
    except Exception as e:
        logger.error(f"Failed to load order file {order_file}: {e}")
        return None


def main(image_folder: str,
         output_path: str = "final_stitched.jpg",
         save_matches: bool = False,
         order_file: str = None) -> Optional[np.ndarray]:
    if not os.path.exists(image_folder):
        logger.error(f"Image folder '{image_folder}' does not exist")
        return None

    custom_order = None
    if order_file:
        custom_order = load_custom_order(order_file)
        if custom_order is None:
            return None

    stitcher = StandaloneImageStitcher(
        image_folder=image_folder,
        output_path=output_path,
        save_matches=save_matches,
        custom_order=custom_order
    )

    final_image = stitcher.process_all()

    if final_image is not None:
        logger.success("Vertical long image stitching successful!")
    else:
        logger.error("Stitching failed, please check image quality and overlap areas.")

    return final_image


if __name__ == "__main__":
    print("Simplified Vertical Long Image Stitching Tool")
    print("=" * 50)
    image_folder = "./imgs"
    output_path = "./output"
    save_matches = True

    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        save_matches = sys.argv[3].lower() in ('true', '1', 'yes', 'on')

    print(f"Image folder: {image_folder}")
    print(f"Output path: {output_path}")
    print(f"Save matches: {save_matches}")
    print("=" * 50)

    result = main(
        image_folder=image_folder,
        output_path=output_path,
        save_matches=save_matches,
        order_file=None
    )

    if result is not None:
        print("\n[SUCCESS] Stitching successful!")
        print(f"Output result: {output_path}")
    else:
        print("\n[ERROR] Stitching failed!")
        print("Please check:")
        print("1. Image folder exists and contains image files")
        print("2. Images have sufficient overlap areas")
        print("3. Image quality and resolution are appropriate")