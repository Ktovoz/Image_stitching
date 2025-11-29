"""
Author: Ktovoz
Date: 2025.11.28
"""

from typing import List, Tuple, Dict, Optional

import numpy as np
from loguru import logger

from .common_utils import ImageDimensions, ArrayAnalyzer


class CropAnalyzer:
    @staticmethod
    def analyze_density_based(y_positions: List[float], img_height: int, window_size: int = 50) -> Tuple[int, float]:
        y_positions = np.array(y_positions)
        best_split = img_height // 2
        min_density = float('inf')

        search_range = range(window_size, img_height - window_size)
        for split_point in search_range:
            nearby_points = ArrayAnalyzer.count_nearby_points(y_positions, split_point, window_size)
            if nearby_points < min_density:
                min_density = nearby_points
                best_split = split_point

        return best_split, min_density

    @staticmethod
    def analyze_gap_based(y_positions: List[float], img_height: int) -> Optional[int]:
        y_positions = np.array(sorted(y_positions))
        gaps = []
        for i in range(len(y_positions) - 1):
            gap = y_positions[i + 1] - y_positions[i]
            if gap > 10:
                gaps.append((gap, y_positions[i], y_positions[i + 1]))

        if gaps:
            max_gap, y1, y2 = max(gaps, key=lambda x: x[0])
            return int((y1 + y2) / 2)
        return None

    @staticmethod
    def analyze_quantile_based(y_positions: List[float], is_top: bool) -> int:
        y_positions = np.array(y_positions)
        if is_top:
            q3 = np.percentile(y_positions, 75)
            return int(q3 + 20)
        else:
            q1 = np.percentile(y_positions, 25)
            return int(q1 - 20)


class CropPositionFinder:
    @staticmethod
    def find_optimal_crop_position(y_positions: List[float], img_height: int, is_top: bool = True) -> int:
        if not y_positions or len(y_positions) < 3:
            return 0 if is_top else img_height

        y_positions = np.array(sorted(y_positions))
        window_size = min(50, img_height // 10)

        best_split, min_density = CropAnalyzer.analyze_density_based(
            y_positions, img_height, window_size)

        gap_result = CropAnalyzer.analyze_gap_based(y_positions, img_height)
        if gap_result and window_size < gap_result < img_height - window_size:
            best_split = gap_result

        quantile_result = CropAnalyzer.analyze_quantile_based(y_positions, is_top)
        if window_size < quantile_result < img_height - window_size:
            quantile_density = ArrayAnalyzer.count_nearby_points(y_positions, quantile_result, window_size)
            if quantile_density <= min_density:
                best_split = quantile_result

        buffer = 30
        result = max(buffer, min(best_split, img_height - buffer))
        return result


class MatchInfoAnalyzer:
    @staticmethod
    def get_match_info_for_pair(match_info: Dict, key: Tuple[int, int]) -> Tuple:
        return match_info.get(key, ([], [], [], -1, None))

    @staticmethod
    def has_valid_match_info(match_data: Tuple) -> bool:
        matches, img1_y_positions, img2_y_positions, order, homography = match_data
        return (img1_y_positions and img2_y_positions and order != 0 and
                len(img1_y_positions) >= 5 and len(img2_y_positions) >= 5)


class ImageCropper:
    @staticmethod
    def should_crop_with_match_info(img_y_positions: List[float], order: int) -> bool:
        return (img_y_positions and order != 0 and len(img_y_positions) >= 5)

    @staticmethod
    def process_image_with_next_match(img: np.ndarray, img_name: str,
                                      match_info: Tuple, img_index: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        matches, img1_y_positions, img2_y_positions, order, homography = match_info
        img_height = ImageDimensions.get_height(img)

        if ImageCropper.should_crop_with_match_info(img1_y_positions, order) and order == -1:
            crop_bottom = CropPositionFinder.find_optimal_crop_position(
                img1_y_positions, img_height, is_top=False)
            cropped_img = img[:crop_bottom, :]
            crop_info = (0, crop_bottom)
            logger.info(f"  [Cropped] {img_name}: Keeping 0-{crop_bottom}   (original{img_height})")
            return cropped_img, crop_info

        return img, (0, img_height)

    @staticmethod
    def process_image_with_prev_match(img: np.ndarray, img_name: str,
                                      match_info: Tuple, img_index: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        matches, img1_y_positions, img2_y_positions, order, homography = match_info
        img_height = ImageDimensions.get_height(img)

        if ImageCropper.should_crop_with_match_info(img2_y_positions, order) and order == -1:
            crop_top = CropPositionFinder.find_optimal_crop_position(
                img2_y_positions, img_height, is_top=True)
            cropped_img = img[crop_top:, :]
            crop_info = (crop_top, img_height)
            logger.info(f"  [Cropped] {img_name}: Keeping {crop_top}-{img_height}   (original{img_height})")
            return cropped_img, crop_info

        return img, (0, img_height)
