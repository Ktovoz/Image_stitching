"""
垂直长图拼接工具 - 基于SIFT特征点的智能图像拼接

Author: Ktovoz
Date: 2025-11-28

基本原理：
1. SIFT特征提取: 使用OpenCV的SIFT算法提取每张图像的关键点和描述符
2. 特征匹配: 使用FLANN匹配器对相邻图像的特征点进行匹配
3. RANSAC过滤: 应用RANSAC算法去除误匹配，提高匹配质量
4. 锚点定位: 基于匹配点的中位数位置确定最佳拼接锚点
5. 智能裁剪: 根据锚点位置对图像进行智能裁剪，避免重叠
6. 图像融合: 将裁剪后的图像垂直拼接成最终长图

特点：
- 自动检测图像顺序（文件名排序或自定义顺序）
- 基于特征点的智能拼接，避免简单的垂直堆叠
- 生成特征匹配可视化，帮助调试拼接质量
- 支持多种图像格式（jpg, jpeg, png）
- 详细的进度反馈和错误处理

使用场景：
- 手机截图拼接成长图
- 文档扫描图片的垂直拼接
- 网页长截图的自动化拼接
"""


import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class StitchingConfig:
    sift_nfeatures: int = 2000
    sift_contrast_threshold: float = 0.015
    sift_edge_threshold: int = 20

    flann_index_kdtree: int = 1
    flann_trees: int = 4
    flann_search_checks: int = 32

    ransac_threshold: float = 3.0
    ransac_max_iters: int = 50

    ratio_test_threshold: float = 0.7
    min_matches_for_ransac: int = 4

    gaussian_kernel_size: Tuple[int, int] = (3, 3)
    gaussian_sigma: float = 0.6

    x_quality_normalizer: float = 30.0
    match_density_normalizer: int = 10
    distribution_quality_divisor: float = 1.0

    fusion_height_ratio: float = 0.4
    max_fusion_height: int = 300
    min_fusion_height: int = 80
    
    viz_max_width: int = 800
    viz_max_height: int = 600
    viz_match_radius: int = 5
    viz_matches_to_show: int = 150
    
    min_features_for_validation: int = 5
    min_matches_for_matching: int = 5

    def get_edge_heights(self, image_height: int) -> List[int]:
        """根据图像高度动态计算边缘高度，无需硬编码"""
        base_values = [6, 4, 3, 2]
        return [min(1200, image_height // div) for div in base_values]

    def calculate_fusion_height(self, edge_height: int, match_count: int) -> int:
        """根据匹配点数量动态计算融合高度，简化配置参数"""
        optimal_height = max(self.min_fusion_height,
                           min(int(edge_height * self.fusion_height_ratio),
                               self.max_fusion_height))
        
        
        if match_count > 50:
            
            return int(optimal_height * 1.2)
        elif match_count > 20:
            
            return optimal_height
        else:
            
            return max(60, int(optimal_height * 0.8))


class ImageProcessor:
    @staticmethod
    def preprocess_for_sift(img: np.ndarray, config: StitchingConfig) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, config.gaussian_kernel_size, config.gaussian_sigma)


class FeatureExtractor:
    def __init__(self, config: StitchingConfig):
        self.sift_detector = cv2.SIFT_create(
            nfeatures=config.sift_nfeatures,
            contrastThreshold=config.sift_contrast_threshold,
            edgeThreshold=config.sift_edge_threshold
        )

    def extract_features(self, img: np.ndarray, config: StitchingConfig) -> Tuple[List, Optional[np.ndarray]]:
        preprocessed = ImageProcessor.preprocess_for_sift(img, config)
        try:
            keypoints, descriptors = self.sift_detector.detectAndCompute(preprocessed, None)
            return keypoints, descriptors
        except:
            return [], None


class FeatureMatcher:
    def __init__(self, config: StitchingConfig):
        self.flann_matcher = cv2.FlannBasedMatcher(
            dict(algorithm=config.flann_index_kdtree, trees=config.flann_trees),
            dict(checks=config.flann_search_checks)
        )
        self.ratio_test_threshold = config.ratio_test_threshold

    def match_features(self, des1: np.ndarray, des2: np.ndarray) -> List:
        if des1 is None or des2 is None or des1.shape[0] == 0 or des2.shape[0] == 0:
            return []

        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
        if des2.dtype != np.float32:
            des2 = des2.astype(np.float32)

        try:
            matches = self.flann_matcher.knnMatch(des1, des2, k=2)

            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_test_threshold * n.distance:
                        good_matches.append(m)

            return good_matches

        except:
            return []


class RansacFilter:
    def __init__(self, config: StitchingConfig):
        self.ransac_threshold = config.ransac_threshold
        self.ransac_max_iters = config.ransac_max_iters
        self.min_matches_for_ransac = config.min_matches_for_ransac

    def apply_ransac(self, kp1: List, kp2: List, matches: List) -> List:
        if len(matches) < self.min_matches_for_ransac:
            return matches

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        try:
            H, mask = cv2.findHomography(
                dst_pts, src_pts, cv2.RANSAC,
                self.ransac_threshold, self.ransac_max_iters
            )

            if H is not None:
                filtered_matches = []
                for i, m in enumerate(matches):
                    if mask[i][0] == 1:
                        filtered_matches.append(m)

                return filtered_matches
            else:
                return matches

        except:
            return matches


class CompleteDebugStitcher:
    def __init__(self, config: StitchingConfig = None):
        self._image_data = []
        self.config = config or StitchingConfig()

        self.feature_extractor = FeatureExtractor(self.config)
        self.feature_matcher = FeatureMatcher(self.config)
        self.ransac_filter = RansacFilter(self.config)

    def _extract_and_validate_features(self, region1: np.ndarray, region2: np.ndarray, min_features: int = None):
        if min_features is None:
            min_features = self.config.min_features_for_validation
            
        kp1, des1 = self.feature_extractor.extract_features(region1, self.config)
        kp2, des2 = self.feature_extractor.extract_features(region2, self.config)

        
        if des1 is None or des2 is None or des1.shape[0] == 0 or des2.shape[0] == 0:
            return None, None, None, None

        
        if len(kp1) < min_features or len(kp2) < min_features:
            return None, None, None, None

        return kp1, des1, kp2, des2

    def _match_and_filter_features(self, des1: np.ndarray, des2: np.ndarray, kp1: List, kp2: List, min_matches: int = None):
        if min_matches is None:
            min_matches = self.config.min_matches_for_matching
            
        
        good_matches = self.feature_matcher.match_features(des1, des2)

        
        if len(good_matches) < min_matches:
            return []

        
        filtered_matches = self.ransac_filter.apply_ransac(kp1, kp2, good_matches)

        if len(filtered_matches) < min_matches:
            return []

        return filtered_matches

    def _detect_overlap_unified(self, img1: np.ndarray, img2: np.ndarray, mode: str = 'enhanced', step_num: int = 0):
        height1, width1 = img1.shape[:2]

        if mode == 'enhanced':
            
            edge_heights = self.config.get_edge_heights(height1)
            return self._enhanced_overlap_detection(img1, img2, edge_heights)

        elif mode == 'median_anchor':
            
            return self._median_anchor_detection(img1, img2, step_num)

        return None

    def _extract_image_regions(self, img1: np.ndarray, img2: np.ndarray, 
                              extraction_type: str = 'overlap', edge_height: int = None) -> Tuple[np.ndarray, np.ndarray]:
        height1, _ = self._validate_and_get_image_size(img1)
        height2, _ = self._validate_and_get_image_size(img2)
        
        if extraction_type == 'overlap' and edge_height is not None:
            return img1[height1 - edge_height:, :], img2[:edge_height, :]
        elif extraction_type == 'anchor':
            return img1[height1 // 2:, :], img2[:height2 // 2, :]
        else:
            raise ValueError(f"Invalid extraction type: {extraction_type}")

    def _enhanced_overlap_detection(self, img1: np.ndarray, img2: np.ndarray, edge_heights: List[int]):
        """Enhanced overlap detection implementation"""
        best_overlap = None
        best_score = 0
        min_matches = self.config.min_matches_for_ransac

        for edge_height in edge_heights:
            img1_bottom, img2_top = self._extract_image_regions(img1, img2, 'overlap', edge_height)

            kp1, des1 = self.feature_extractor.extract_features(img1_bottom, self.config)
            kp2, des2 = self.feature_extractor.extract_features(img2_top, self.config)
            
            if not (des1 is None or des2 is None or 
                    des1.shape[0] == 0 or des2.shape[0] == 0 or
                    len(kp1) < min_matches or len(kp2) < min_matches):
                
                good_matches = self._match_and_filter_features(des1, des2, kp1, kp2, None)

                if not good_matches:
                    continue

                height1, _ = self._validate_and_get_image_size(img1)
                overlap_info = self._calculate_overlap_metrics(kp1, kp2, good_matches, edge_height, height1)

                if overlap_info['score'] > best_score:
                    best_score = overlap_info['score']
                    best_overlap = overlap_info

        return best_overlap

    def _median_anchor_detection(self, img1: np.ndarray, img2: np.ndarray, step_num: int):

        print(f"--- Step {step_num}: Median Anchor Stitching ---")

        img1_bottom, img2_top = self._extract_image_regions(img1, img2, 'anchor')

        kp1, des1, kp2, des2 = self._extract_and_validate_features(img1_bottom, img2_top, None)
        if des1 is None:
            return None

        good_matches = self._match_and_filter_features(des1, des2, kp1, kp2, self.config.min_matches_for_ransac)
        if not good_matches:
            return None

        
        height1, width1 = self._validate_and_get_image_size(img1)
        _, width2 = self._validate_and_get_image_size(img2)
        
        anchor_info = self._calculate_median_anchor(good_matches, kp1, kp2, height1)

        median_y1 = anchor_info['median_y1']
        median_y2 = anchor_info['median_y2']

        img1_cropped = img1[:int(median_y1), :]
        img2_cropped = img2[int(median_y2):, :]

        new_height = img1_cropped.shape[0] + img2_cropped.shape[0]
        new_width = max(width1, width2)

        result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        result[:img1_cropped.shape[0], :width1] = img1_cropped
        result[img1_cropped.shape[0]:img1_cropped.shape[0] + img2_cropped.shape[0], :width2] = img2_cropped

        return {
            'result': result,
            'median_y1': median_y1,
            'median_y2': median_y2,
            'crop_y1': int(median_y1),
            'crop_y2': int(median_y2),
            'match_count': len(good_matches),
            'y_coords_img1': anchor_info['y_coords_img1'],
            'y_coords_img2': anchor_info['y_coords_img2']
        }

    def _calculate_overlap_metrics(self, kp1: List, kp2: List, good_matches: List,
                                  edge_height: int, height1: int):

        y_offsets = []
        x_offsets = []
        global_y1_offset = height1 - edge_height

        for match in good_matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt

            global_y1 = pt1[1] + global_y1_offset
            global_y2 = pt2[1]

            y_offsets.append(global_y1 - global_y2)
            x_offsets.append(pt1[0] - pt2[0])

        y_offsets = np.array(y_offsets)
        x_offsets = np.array(x_offsets)

        y_std = np.std(y_offsets)
        abs_y_mean = abs(np.mean(y_offsets))
        abs_x_mean = abs(np.mean(x_offsets))
        
        y_consistency = 1.0 / (1.0 + y_std / max(abs_y_mean, 1))
        x_quality = 1.0 / (1.0 + abs_x_mean / self.config.x_quality_normalizer)
        match_density = min(1.0, len(good_matches) / max(edge_height // self.config.match_density_normalizer, 5))

        distribution_quality = 0.5 if len(y_offsets) <= 3 else (
            1.0 / (1.0 + max(np.max(y_offsets) - np.min(y_offsets), 
                             np.max(x_offsets) - np.min(x_offsets)) / 
               (edge_height * self.config.distribution_quality_divisor)))

        total_score = len(good_matches) * y_consistency * x_quality * match_density * distribution_quality

        
        fusion_height = int(self.config.calculate_fusion_height(edge_height, len(good_matches)))

        return {
            'match_count': len(good_matches),
            'edge_height': edge_height,
            'good_matches': good_matches,
            'kp1': kp1,
            'score': total_score,
            'fusion_height': fusion_height,
            'fusion_start_y_img1': height1 - fusion_height
        }

    def _calculate_median_anchor(self, good_matches: List, kp1: List, kp2: List, height1: int):

        half_height = height1 // 2
        y_coords_img1 = []
        y_coords_img2 = []

        for match in good_matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt

            y_coords_img1.append(pt1[1] + half_height)
            y_coords_img2.append(pt2[1])

        median_y1 = np.median(y_coords_img1)
        median_y2 = np.median(y_coords_img2)

        return {
            'median_y1': median_y1,
            'median_y2': median_y2,
            'match_count': len(good_matches),
            'good_matches': good_matches,
            'kp1': kp1,
            'kp2': kp2,
            'y_coords_img1': y_coords_img1,
            'y_coords_img2': y_coords_img2
        }

    def load_images(self, image_paths: List[str]):
        self._image_data = []

        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                self._image_data.append((img, path))

        if len(self._image_data) < len(image_paths):
            print(f"[ERROR] Image loading failed: {len(self._image_data)}/{len(image_paths)} images loaded")

    @property
    def images(self) -> List[np.ndarray]:

        return [data[0] for data in self._image_data]

    @property
    def image_paths(self) -> List[str]:

        return [data[1] for data in self._image_data]

    def _validate_and_get_image_size(self, img: np.ndarray) -> Tuple[int, int]:
        if img is None or img.size == 0:
            raise ValueError("Image is empty or invalid")
        
        if len(img.shape) < 2:
            raise ValueError("Insufficient image dimensions")
        
        height, width = img.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image dimensions: {height}x{width}")
        
        return height, width

    def get_image_path(self, index: int) -> str:

        if 0 <= index < len(self._image_data):
            return self._image_data[index][1]
        return None

    def get_image_info(self, index: int) -> Tuple[np.ndarray, str]:

        if 0 <= index < len(self._image_data):
            return self._image_data[index]
        return None, None

    def visualize_feature_matches(self, img1, img2, save_path, img1_position='top'):
        
        height1, width1 = self._validate_and_get_image_size(img1)
        height2, width2 = self._validate_and_get_image_size(img2)

        
        roi1 = img1[height1 // 2:, :] if img1_position == 'top' else img1[:height1 // 2, :]
        roi2 = img2[:height2 // 2, :] if img1_position == 'top' else img2[height2 // 2:, :]

        
        kp1, des1, kp2, des2 = self._extract_and_validate_features(roi1, roi2, None)
        if des1 is None:
            return 0

        
        good_matches = self._match_and_filter_features(des1, des2, kp1, kp2, None)
        if not good_matches:
            return 0

        final_matches = sorted(good_matches, key=lambda x: x.distance)[:self.config.viz_matches_to_show]

        
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        
        display_img, display_info = self._create_image_layout(
            img1_rgb, img2_rgb, (width1, height1), (width2, height2)
        )

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(display_img)

        
        region_info = "Image1 Bottom Half ↔ Image2 Top Half" if img1_position == 'top' else "Image1 Top Half ↔ Image2 Bottom Half"
        ax.set_title(f'Region-Based SIFT+FLANN Feature Matching\n{region_info} - {len(final_matches)} Matches',
                     fontsize=14, fontweight='bold')

        
        self._draw_matches_on_figure(
            ax, final_matches, kp1, kp2, display_info, img1_position, 
            (height1, height2), (width1, width2)
        )

        ax.axhline(y=display_info['display_height1'], color='white', linewidth=4, alpha=0.9)

        
        region_data = {
            'top': {'img1_y_start': display_info['display_height1'] // 2, 'img2_y_start': 0, 
                   'img1_region': "Image 1 (Bottom Half)", 'img2_region': "Image 2 (Top Half)"},
            'bottom': {'img1_y_start': 0, 'img2_y_start': display_info['display_height2'] // 2,
                      'img1_region': "Image 1 (Top Half)", 'img2_region': "Image 2 (Bottom Half)"}
        }
        
        data = region_data[img1_position]
        
        
        for color, y_start, rect_height in [('yellow', data['img1_y_start'], display_info['display_height1'] // 2),
                                           ('cyan', display_info['display_height1'] + data['img2_y_start'], display_info['display_height2'] // 2)]:
            rect = patches.Rectangle((0, y_start), display_img.shape[1], rect_height,
                                   linewidth=3, edgecolor=color, facecolor=color, alpha=0.1)
            ax.add_patch(rect)

        
        ax.text(display_img.shape[1] // 2, display_info['display_height1'] // 2, data['img1_region'],
                ha='center', va='center', color='white', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
        ax.text(display_img.shape[1] // 2, display_info['display_height1'] + display_info['display_height2'] // 2, data['img2_region'],
                ha='center', va='center', color='white', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))

        ax.set_xlim(0, display_img.shape[1])
        ax.set_ylim(display_img.shape[0], 0)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='black')
        plt.close()

        return len(final_matches)

    def _create_image_layout(self, img1_rgb: np.ndarray, img2_rgb: np.ndarray, 
                            size1: Tuple[int, int], size2: Tuple[int, int]) -> Tuple[np.ndarray, dict]:

        width1, height1 = size1
        width2, height2 = size2
        
        max_width, max_height = self.config.viz_max_width, self.config.viz_max_height
        scale = min(max_width / max(width1, width2), max_height / (height1 + height2))
        
        display_width1 = int(width1 * scale)
        display_height1 = int(height1 * scale)
        display_width2 = int(width2 * scale)
        display_height2 = int(height2 * scale)
        
        img1_resized = cv2.resize(img1_rgb, (display_width1, display_height1))
        img2_resized = cv2.resize(img2_rgb, (display_width2, display_height2))
        
        total_height = display_height1 + display_height2
        display_img = np.zeros((total_height, max(display_width1, display_width2), 3), dtype=np.uint8)
        
        display_info = {
            'display_width1': display_width1,
            'display_height1': display_height1,
            'display_width2': display_width2,
            'display_height2': display_height2,
            'scale_x1': scale,
            'scale_y1': scale,
            'scale_x2': scale,
            'scale_y2': scale,
            'start_x1': 0,
            'start_x2': 0
        }
        
        if display_width1 >= display_width2:
            display_info['start_x1'] = 0
            display_info['start_x2'] = (display_width1 - display_width2) // 2
            display_img[:display_height1, :display_width1] = img1_resized
            display_img[display_height1:display_height1 + display_height2,
                       display_info['start_x2']:display_info['start_x2'] + display_width2] = img2_resized
        else:
            display_info['start_x2'] = 0
            display_info['start_x1'] = (display_width2 - display_width1) // 2
            display_img[:display_height1, display_info['start_x1']:display_info['start_x1'] + display_width1] = img1_resized
            display_img[display_height1:display_height1 + display_height2,
                       :display_width2] = img2_resized
        
        return display_img, display_info

    def _draw_matches_on_figure(self, ax, final_matches: List, kp1: List, kp2: List, 
                               display_info: dict, img1_position: str, 
                               heights: Tuple[int, int], widths: Tuple[int, int]):

        height1, height2 = heights

        
        for match in final_matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt

            if img1_position == 'top':
                global_y1 = pt1[1] + height1 // 2
            else:
                global_y1 = pt1[1]

            x1 = pt1[0] * display_info['scale_x1']
            y1 = global_y1 * display_info['scale_y1']

            if img1_position == 'top':
                global_y2 = pt2[1]
            else:
                global_y2 = pt2[1] + height2 // 2

            x2 = pt2[0] * display_info['scale_x2']
            y2 = global_y2 * display_info['scale_y2'] + display_info['display_height1']

            x1 += display_info['start_x1']
            x2 += display_info['start_x2']

            ax.plot([x1, x2], [y1, y2], color='red', alpha=0.8, linewidth=2)

            circle1 = patches.Circle((x1, y1), self.config.viz_match_radius, 
                                   alpha=0.9, linewidth=2, edgecolor='white', facecolor='red')
            circle2 = patches.Circle((x2, y2), self.config.viz_match_radius, 
                                   alpha=0.9, linewidth=2, edgecolor='white', facecolor='red')
            ax.add_patch(circle1)
            ax.add_patch(circle2)

    def stitch_with_median_anchor(self):
        if len(self.images) < 2:
            return None

        print(f"\n=== Starting Median Anchor Stitching ===")
        result = self.images[0].copy()

        for i, img in enumerate(self.images[1:], 1):
            anchor_info = self._detect_overlap_unified(result, img, mode='median_anchor', step_num=i)
            
            
            result = anchor_info['result'] if anchor_info else np.vstack([result, img])

        print(f"Final image size: {result.shape}")
        return result



def stitch_images_from_directory(image_dir: str, output_dir: str = "output",
                                 order_mapping: dict = None, save_matches: bool = True):

    print(f"Vertical Image Stitching Tool")
    print(f"Input: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Save matches: {save_matches}")

    os.makedirs(output_dir, exist_ok=True)

    
    img_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if order_mapping:
        img_files.sort(key=lambda x: order_mapping.get(x, float('inf')))
    else:
        img_files.sort()

    if len(img_files) < 2:
        print(f"[ERROR] Need at least 2 images, found {len(img_files)}")
        return None

    
    img_paths = [os.path.join(image_dir, f) for f in img_files]

    
    stitcher = CompleteDebugStitcher()
    stitcher.load_images(img_paths)

    if len(stitcher.images) < 2:
        print("[ERROR] Not enough valid images")
        return None

    
    if save_matches:
        match_vis_dir = os.path.join(output_dir, "matches")
        os.makedirs(match_vis_dir, exist_ok=True)
        
        for i, (img1, img2) in enumerate(zip(stitcher.images[:-1], stitcher.images[1:]), 1):
            match_path = os.path.join(match_vis_dir, f"matches_{i}_{i+1}.jpg")
            stitcher.visualize_feature_matches(img1, img2, match_path, 'top')

    
    result = stitcher.stitch_with_median_anchor()
    
    
    if result is not None:
        final_path = os.path.join(output_dir, "stitched_image.jpg")
        cv2.imwrite(final_path, result)
        
        print(f"Completed! Size: {result.shape[0]}x{result.shape[1]}")
        print(f"Saved to: {final_path}")
        
        if save_matches:
            print(f"Matches: {output_dir}/matches/")
        
        return final_path
    else:
        print("[ERROR] Stitching failed")
        return None


def main():
    result1 = stitch_images_from_directory("imgs", "output", save_matches=True)
    if result1:
        pass


if __name__ == "__main__":
    main()