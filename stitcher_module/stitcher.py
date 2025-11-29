"""
Author: Ktovoz
Date: 2025.11.28
"""

from typing import List, Dict, Tuple, Optional

import numpy as np
from loguru import logger

from .common_utils import (
    ImageProcessor, ValidationUtils,
    ImageDimensions
)
from .feature_matcher import AKAZEMatcher
from .image_cropper import ImageCropper, CropPositionFinder, MatchInfoAnalyzer
from .output_manager import OutputManager, StitchingInfoManager


class StandaloneImageStitcher:
    def __init__(self,
                 image_folder: str,
                 output_path: str = "final_stitched.jpg",
                 save_matches: bool = False,
                 custom_order: Optional[Dict[str, int]] = None):
        self.image_folder = image_folder
        self.output_base_path = output_path
        self.save_matches = save_matches
        self.custom_order = custom_order or {}

        self.output_paths = OutputManager.create_output_structure(
            output_path, save_matches)

        self.matcher = AKAZEMatcher()
        self.images = []
        self.image_names = []
        self.sorted_indices = []

        self._log_initialization_info()
        self.load_images()
        self.sort_images()

    def _log_initialization_info(self):
        logger.info("Standalone Enhanced Image Stitcher - Initialization completed")

    def load_images(self):
        self.images, self.image_names = ValidationUtils.validate_and_load_images(self.image_folder)

        if not ValidationUtils.validate_image_count(self.images):
            logger.error("Insufficient number of images loaded, unable to proceed with stitching")
            return

    def sort_images(self):
        if self.custom_order:
            self._sort_with_custom_order()
        else:
            self._sort_by_filename()

    def _sort_with_custom_order(self):
        logger.info("Sorting with custom order...")
        order_map = {}
        default_order = len(self.image_names)

        for i, filename in enumerate(self.image_names):
            if filename in self.custom_order:
                order_map[i] = self.custom_order[filename]
                logger.info(f"  {filename} -> Position {self.custom_order[filename]}")
            else:
                order_map[i] = default_order + i

        self.sorted_indices = sorted(range(len(self.image_names)),
                                     key=lambda x: order_map[x])

    def _sort_by_filename(self):
        logger.info("Sorting by filename (natural order)...")
        self.sorted_indices = sorted(range(len(self.image_names)),
                                     key=lambda x: self.image_names[x])

    def get_sorted_images(self) -> List[np.ndarray]:
        return [self.images[i] for i in self.sorted_indices]

    def get_sorted_names(self) -> List[str]:
        return [self.image_names[i] for i in self.sorted_indices]

    def analyze_adjacent_pairs(self, sorted_images: List[np.ndarray],
                               sorted_names: List[str]) -> Dict:
        if len(sorted_images) < 2:
            return {}

        logger.info("Analyzing image matching relationships...")

        match_info = {}
        for i in range(len(sorted_images) - 1):
            self._analyze_single_pair(match_info, i)

        success_count = len([k for k, v in match_info.items() if v[0]])
        logger.info(f"Matching completed: {success_count}/{len(sorted_images) - 1} pairs successful")

        return match_info

    def _analyze_single_pair(self, match_info: Dict, pair_index: int):
        sorted_images = self.get_sorted_images()
        sorted_names = self.get_sorted_names()
        idx1, idx2 = pair_index, pair_index + 1
        img1, img2 = sorted_images[idx1], sorted_images[idx2]

        logger.debug(f"Analyzing image pair: {sorted_names[idx1]} vs {sorted_names[idx2]}")

        matches, homography, order, img1_y_positions, img2_y_positions = \
            self.matcher.match_features(img1, img2)

        if matches and homography is not None:
            if self.save_matches:
                self._save_match_visualization(img1, img2, sorted_names[idx1],
                                               sorted_names[idx2], idx1, idx2)

            match_info[(idx1, idx2)] = (matches, img1_y_positions, img2_y_positions, order, homography)

            if order == -1:
                logger.success(f"{sorted_names[idx1]} at top, {sorted_names[idx2]} at bottom")
            elif order == 1:
                logger.success(f"{sorted_names[idx2]} at top, {sorted_names[idx1]} at bottom")
            else:
                logger.warning("Unable to determine order, will use default sequence for stitching")
                order = -1
                match_info[(idx1, idx2)] = (matches, img1_y_positions, img2_y_positions, order, homography)
        else:
            logger.warning("Image pair matching failed, will stack directly")
            match_info[(idx1, idx2)] = ([], [], [], -1, None)

    def _save_match_visualization(self, img1: np.ndarray, img2: np.ndarray,
                                  name1: str, name2: str, idx1: int, idx2: int):
        kp1, _ = self.matcher.detect_akaze_features(img1)
        kp2, _ = self.matcher.detect_akaze_features(img2)

        matches, _, _, _, _ = self.matcher.match_features(img1, img2)

        if matches:
            vis_img = self.matcher.visualize_matches(
                img1, img2, kp1, kp2, matches, f"{name1} vs {name2}")
            OutputManager.save_match_visualization(
                self.output_paths["match_dir"], vis_img, name1, name2, idx1, idx2)

    def smart_crop_all_images(self, sorted_images: List[np.ndarray],
                              sorted_names: List[str],
                              match_info: Dict) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:

        cropped_images = []
        crop_info = []

        for i in range(len(sorted_images)):
            if i == 0:
                cropped_img, info = self._crop_first_image(match_info, i)
            elif i == len(sorted_images) - 1:
                cropped_img, info = self._crop_last_image(match_info, i)
            else:
                cropped_img, info = self._process_middle_image(
                    sorted_images[i], sorted_names[i], match_info, i)

            cropped_images.append(cropped_img)
            crop_info.append(info)

        return cropped_images, crop_info

    def _crop_first_image(self, match_info: Dict, index: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        sorted_images = self.get_sorted_images()
        sorted_names = self.get_sorted_names()
        next_match_info = MatchInfoAnalyzer.get_match_info_for_pair(match_info, (0, 1))
        return ImageCropper.process_image_with_next_match(
            sorted_images[0], sorted_names[0], next_match_info, 0)

    def _crop_last_image(self, match_info: Dict, index: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        sorted_images = self.get_sorted_images()
        sorted_names = self.get_sorted_names()
        prev_key = (len(sorted_images) - 2, len(sorted_images) - 1)
        prev_match_info = MatchInfoAnalyzer.get_match_info_for_pair(match_info, prev_key)
        return ImageCropper.process_image_with_prev_match(
            sorted_images[-1], sorted_names[-1], prev_match_info, index)

    def _process_middle_image(self, img: np.ndarray, img_name: str,
                              match_info: Dict, index: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        prev_key = (index - 1, index)
        next_key = (index, index + 1)

        img_height = ImageDimensions.get_height(img)
        crop_top = 0
        crop_bottom = img_height
        should_crop = False

        prev_match_info = MatchInfoAnalyzer.get_match_info_for_pair(match_info, prev_key)
        if MatchInfoAnalyzer.has_valid_match_info(prev_match_info):
            _, img1_y_positions, img2_y_positions, order, _ = prev_match_info
            if order == -1:
                crop_top = CropPositionFinder.find_optimal_crop_position(
                    img2_y_positions, img_height, is_top=True)
                should_crop = True

        next_match_info = MatchInfoAnalyzer.get_match_info_for_pair(match_info, next_key)
        if MatchInfoAnalyzer.has_valid_match_info(next_match_info):
            _, img1_y_positions, img2_y_positions, order, _ = next_match_info
            if order == -1:
                crop_bottom = CropPositionFinder.find_optimal_crop_position(
                    img1_y_positions, img_height, is_top=False)
                should_crop = True

        if should_crop and crop_top < crop_bottom:
            cropped_img = img[crop_top:crop_bottom, :]
            logger.info(f"  [Cropped] {img_name}: Keeping {crop_top}-{crop_bottom} (original {img_height})")
            return cropped_img, (crop_top, crop_bottom)
        else:
            return img, (0, img.shape[0])

    def unify_image_widths(self, cropped_images: List[np.ndarray]) -> List[np.ndarray]:
        unified_images = ImageProcessor.normalize_image_size(cropped_images)
        return unified_images

    def create_final_stitch(self, sorted_images: List[np.ndarray], sorted_names: List[str],
                            match_info: Dict) -> Optional[np.ndarray]:
        if not ValidationUtils.validate_image_count(sorted_images):
            return None

        logger.info("Performing intelligent cropping...")
        total_original_height = sum(ImageDimensions.get_height(img) for img in sorted_images)
        logger.info(f"Total original height: {total_original_height}px")

        cropped_images, crop_info = self.smart_crop_all_images(sorted_images, sorted_names, match_info)
        unified_images = self.unify_image_widths(cropped_images)
        final_image = self._perform_vertical_stitching(unified_images, sorted_names)

        if final_image is None:
            return None

        self._finalize_output(final_image, crop_info, total_original_height, sorted_names)
        return final_image

    def _calculate_saving_stats(self, final_image: np.ndarray, total_original_height: int) -> Dict[str, int]:
        final_height = final_image.shape[0]
        height_saved = total_original_height - final_height
        saving_ratio = int((height_saved / total_original_height) * 100)
        return {
            "final_height": final_height,
            "height_saved": height_saved,
            "saving_ratio": saving_ratio
        }

    def _perform_vertical_stitching(self, unified_images: List[np.ndarray],
                                    sorted_names: List[str]) -> Optional[np.ndarray]:
        logger.info("Starting vertical stitching...")

        try:
            final_image = unified_images[0]
            for i in range(1, len(unified_images)):
                final_image = np.vstack([final_image, unified_images[i]])
            return final_image
        except Exception as e:
            logger.error(f"Vertical stitching failed: {e}")
            return None

    def _finalize_output(self, final_image: np.ndarray, crop_info: List[Tuple[int, int]],
                         total_original_height: int, sorted_names: List[str]) -> None:
        stats = self._calculate_saving_stats(final_image, total_original_height)

        logger.success("Stitching completed!")
        logger.info(f"Final dimensions: {final_image.shape[1]}x{final_image.shape[0]}px, saved {stats['saving_ratio']}%")

        OutputManager.ensure_output_directories(self.output_paths)
        OutputManager.save_final_image(self.output_paths["final_image_path"], final_image)

        StitchingInfoManager.save_stitching_info(
            self.output_paths, self.image_folder, self.save_matches,
            self.custom_order, sorted_names, self.images, crop_info,
            total_original_height, stats["final_height"],
            stats["height_saved"], stats["saving_ratio"])

    def process_all(self) -> Optional[np.ndarray]:
        logger.info("Starting image stitching...")

        if not ValidationUtils.validate_image_count(self.images):
            return None

        sorted_images = self.get_sorted_images()
        sorted_names = self.get_sorted_names()

        match_info = self.analyze_adjacent_pairs(sorted_images, sorted_names)

        final_image = self.create_final_stitch(sorted_images, sorted_names, match_info)

        logger.info("Image stitching completed!")
        return final_image
