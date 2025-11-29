"""
Author: Ktovoz
Date: 2025.11.28
"""

from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

from .common_utils import (
    FeatureDetector, GeometryValidator,
    PositionAnalyzer, ValidationUtils,
    GeometryProcessor,
    DEFAULT_IMAGE_NAME
)


class AKAZEMatcher:
    def __init__(self):
        self.akaze = cv2.AKAZE_create()
        self.matcher = FeatureDetector.create_bf_matcher()

    def detect_akaze_features(self, img: np.ndarray, img_name: str = DEFAULT_IMAGE_NAME) -> Tuple[List, np.ndarray]:
        return FeatureDetector.detect_akaze_features(img, img_name)

    def analyze_match_regions(self, kp1: List, kp2: List, matches: List,
                              img1_shape: Tuple, img2_shape: Tuple) -> Tuple[int, float, float, List, List]:
        if len(matches) < 5:
            return 0, 0.0, 0.0, [], []

        h1, w1 = img1_shape[:2]
        h2, w2 = img2_shape[:2]

        img1_y_positions = PositionAnalyzer.extract_keypoint_positions(kp1, matches, query_idx=True)
        img2_y_positions = PositionAnalyzer.extract_keypoint_positions(kp2, matches, query_idx=False)

        analysis1 = PositionAnalyzer.analyze_vertical_distribution(img1_y_positions, h1)
        analysis2 = PositionAnalyzer.analyze_vertical_distribution(img2_y_positions, h2)

        avg_y1, avg_y2 = analysis1["mean"], analysis2["mean"]
        logger.info(
            f"Region analysis: Image1({analysis1['upper_count']}/{analysis1['lower_count']}) vs Image2({analysis2['upper_count']}/{analysis2['lower_count']})")
        logger.info(f"  Average positions: Y1={avg_y1:.1f}, Y2={avg_y2:.1f}")

        order = PositionAnalyzer.determine_stitching_order(analysis1, analysis2)

        if order == 0 and len(matches) >= 8:
            if avg_y1 > h1 / 2 and avg_y2 < h2 / 2:
                return -1, avg_y1, avg_y2, img1_y_positions, img2_y_positions
            elif avg_y1 < h1 / 2 and avg_y2 > h2 / 2:
                return 1, avg_y1, avg_y2, img1_y_positions, img2_y_positions

        return order, avg_y1, avg_y2, img1_y_positions, img2_y_positions

    def match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[
        List, np.ndarray, int, List[float], List[float]]:
        logger.info("Performing feature matching...")

        kp1, desc1 = self.detect_akaze_features(img1, f"{DEFAULT_IMAGE_NAME}1")
        kp2, desc2 = self.detect_akaze_features(img2, f"{DEFAULT_IMAGE_NAME}2")

        if not FeatureDetector.validate_features(kp1, desc1):
            logger.error("Image1 feature detection failed")
            return [], None, 0, [], []
        if not FeatureDetector.validate_features(kp2, desc2):
            logger.error("Image2 feature detection failed")
            return [], None, 0, [], []

        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:max(len(matches) // 2, len(matches) // 3)]

        if ValidationUtils.validate_match_count(good_matches):
            src_pts = GeometryProcessor.extract_keypoint_coordinates(kp1, good_matches, use_query=True)
            dst_pts = GeometryProcessor.extract_keypoint_coordinates(kp2, good_matches, use_query=False)

            homography, mask = GeometryValidator.find_homography(src_pts, dst_pts)

            if GeometryValidator.validate_homography(homography):
                inlier_matches = GeometryValidator.filter_inliers(mask, good_matches)

                order, avg_y1, avg_y2, img1_y_positions, img2_y_positions = self.analyze_match_regions(
                    kp1, kp2, inlier_matches, img1.shape, img2.shape)

                return inlier_matches, homography, order, img1_y_positions, img2_y_positions
            else:
                logger.error("Unable to compute valid homography matrix")
        else:
            logger.error("Insufficient match points for geometric validation")

        return [], None, 0, [], []

    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray,
                          kp1: List, kp2: List, matches: List,
                          title: str) -> np.ndarray:
        match_image = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        h, w = match_image.shape[:2]
        info_height = 80
        info_bar = np.zeros((info_height, w, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(info_bar, f"AKAZE Matching - {title}", (10, 25), font, 0.8, (255, 255, 255), 2)
        cv2.putText(info_bar, f"Matches: {len(matches)}", (10, 50), font, 0.6, (0, 255, 255), 2)
        cv2.putText(info_bar, f"Keypoints: {len(kp1)} / {len(kp2)}", (10, 70), font, 0.5, (255, 255, 0), 2)

        result = np.vstack([info_bar, match_image])

        if w > 1600:
            scale = 1600 / w
            new_h = int((h + info_height) * scale)
            result = cv2.resize(result, (1600, new_h))

        return result
