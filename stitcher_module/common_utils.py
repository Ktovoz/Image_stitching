"""
Author: Ktovoz
Date: 2025.11.28
"""

import os
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from loguru import logger

IMAGE_EXTENSIONS: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
DEFAULT_IMAGE_NAME: str = "图像"


def is_image_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


class ImageDimensions:
    @staticmethod
    def get_height(img) -> int:
        return img.shape[0]

    @staticmethod
    def get_size(img) -> tuple:
        return (img.shape[0], img.shape[1])


class GeometryProcessor:
    @staticmethod
    def extract_keypoint_coordinates(keypoints, matches, use_query=True):
        coords = []
        for match in matches:
            kp = keypoints[match.queryIdx if use_query else match.trainIdx]
            coords.append(kp.pt)
        return np.float32(coords).reshape(-1, 1, 2)


class FeatureStatistics:
    pass


class ArrayAnalyzer:
    @staticmethod
    def count_nearby_points(y_positions, split_point, window_size):
        return np.sum(np.abs(y_positions - split_point) < window_size)


class ImageProcessor:
    @staticmethod
    def validate_image(img: np.ndarray, img_name: str = DEFAULT_IMAGE_NAME) -> bool:
        if img is None:
            logger.error(f"{img_name} is empty or failed to load")
            return False
        if len(img.shape) < 2:
            logger.error(f"{img_name} insufficient dimensions: {img.shape}")
            return False
        return True

    @staticmethod
    def ensure_gray_image(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3 and img.shape[2] >= 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray
        return img

    @staticmethod
    def normalize_image_size(images: List[np.ndarray], target_width: Optional[int] = None) -> List[np.ndarray]:
        if not images:
            return []

        if target_width is None:
            target_width = min(img.shape[1] for img in images)

        normalized = []
        for img in images:
            h, w = ImageDimensions.get_size(img)
            if w != target_width:
                resized = cv2.resize(img, (target_width, h))
                normalized.append(resized)
            else:
                normalized.append(img)

        return normalized


class FeatureDetector:
    @staticmethod
    def detect_akaze_features(img: np.ndarray, img_name: str = DEFAULT_IMAGE_NAME) -> Tuple[List, np.ndarray]:
        if not ImageProcessor.validate_image(img, img_name):
            return [], None

        gray = ImageProcessor.ensure_gray_image(img)
        akaze = cv2.AKAZE_create()
        keypoints, descriptors = akaze.detectAndCompute(gray, None)

        return keypoints, descriptors

    @staticmethod
    def create_bf_matcher(norm_type: int = cv2.NORM_L2, cross_check: bool = True):
        return cv2.BFMatcher(norm_type, crossCheck=cross_check)

    @staticmethod
    def validate_features(keypoints: List, descriptors: np.ndarray) -> bool:
        if not keypoints or len(keypoints) < 5:
            logger.error(f"Insufficient keypoints: {len(keypoints) if keypoints else 0} < {5}")
            return False
        if descriptors is None or descriptors.shape[1] < 1:
            logger.error(f"Invalid descriptors: {descriptors.shape if descriptors is not None else None}")
            return False
        return True


class GeometryValidator:
    @staticmethod
    def find_homography(src_pts: np.ndarray, dst_pts: np.ndarray,
                        method: int = cv2.RANSAC, ransac_thresh: float = 3.0) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None, None

        try:
            homography, mask = cv2.findHomography(src_pts, dst_pts, method, ransac_thresh)
            return homography, mask
        except Exception as e:
            logger.error(f"Homography matrix computation failed: {e}")
            return None, None

    @staticmethod
    def validate_homography(homography: np.ndarray) -> bool:
        if homography is None:
            return False
        if np.isnan(homography).any() or np.isinf(homography).any():
            return False
        det = np.linalg.det(homography)
        return abs(det) > 1e-10

    @staticmethod
    def filter_inliers(mask: np.ndarray, matches: List) -> List:
        if mask is None:
            return matches
        return [matches[i] for i in range(len(matches)) if mask.ravel()[i] == 1]


class PositionAnalyzer:
    @staticmethod
    def extract_keypoint_positions(keypoints: List, matches: List,
                                   query_idx: bool = True) -> List[float]:
        positions = []
        for match in matches:
            if query_idx:
                pt = keypoints[match.queryIdx].pt
            else:
                pt = keypoints[match.trainIdx].pt
            positions.append(pt[1])
        return positions

    @staticmethod
    def analyze_vertical_distribution(y_positions: List[float], img_height: int) -> Dict[str, Any]:
        if not y_positions:
            return {"error": "empty_positions"}

        y_array = np.array(y_positions)
        return {
            "count": len(y_positions),
            "min": float(np.min(y_array)),
            "max": float(np.max(y_array)),
            "mean": float(np.mean(y_array)),
            "median": float(np.median(y_array)),
            "std": float(np.std(y_array)),
            "upper_quartile": float(np.percentile(y_array, 75)),
            "lower_quartile": float(np.percentile(y_array, 25)),
            "mid_point": img_height / 2,
            "upper_count": int(np.sum(y_array < img_height / 2)),
            "lower_count": int(np.sum(y_array >= img_height / 2))
        }

    @staticmethod
    def determine_stitching_order(analysis1: Dict, analysis2: Dict) -> int:
        if "error" in analysis1 or "error" in analysis2:
            return 0

        if (analysis1["upper_count"] < analysis1["lower_count"] and
                analysis2["upper_count"] > analysis2["lower_count"]):
            return -1
        elif (analysis1["upper_count"] > analysis1["lower_count"] and
              analysis2["upper_count"] < analysis2["lower_count"]):
            return 1
        else:
            return 0


class ValidationUtils:
    @staticmethod
    def validate_and_load_images(folder_path: str) -> Tuple[List[np.ndarray], List[str]]:
        if not os.path.exists(folder_path):
            logger.error(f"Folder does not exist: {folder_path}")
            return [], []

        image_files = [f for f in os.listdir(folder_path)
                       if is_image_file(f)]

        if len(image_files) < 2:
            logger.error(f"Insufficient image files: found{len(image_files)} files, need at least {2} files")
            return [], []

        images = []
        names = []

        for filename in sorted(image_files):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if ImageProcessor.validate_image(img, filename):
                images.append(img)
                names.append(filename)
            else:
                logger.warning(f"Skipping invalid image: {filename}")

        return images, names

    @staticmethod
    def validate_match_count(matches: List) -> bool:
        return len(matches) >= 4

    @staticmethod
    def validate_image_count(images: List) -> bool:
        return len(images) >= 2
