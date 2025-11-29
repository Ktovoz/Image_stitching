"""
Author: Ktovoz
Date: 2025.11.28
"""

import json
import os
from typing import List, Dict, Tuple

import cv2

from .config import *


class OutputManager:
    @staticmethod
    def create_output_structure(output_base_path: str, save_matches: bool = True) -> Dict[str, str]:
        base_dir, base_name = OutputManager._parse_output_path(output_base_path)

        output_dir = os.path.join(base_dir, f"{base_name}{OUTPUT_SUFFIX}")
        final_image_path = os.path.join(output_dir, f"{base_name}.jpg")
        match_dir = os.path.join(output_dir, MATCHES_DIR_NAME) if save_matches else None
        info_file = os.path.join(output_dir, INFO_FILE_NAME)

        return {
            "output_dir": output_dir,
            "final_image_path": final_image_path,
            "match_dir": match_dir,
            "info_file": info_file,
            "base_name": base_name
        }

    @staticmethod
    def _parse_output_path(output_base_path: str) -> Tuple[str, str]:
        if os.path.splitext(output_base_path)[1]:
            base_dir = os.path.dirname(output_base_path) or "."
            base_name = os.path.splitext(os.path.basename(output_base_path))[0]
        else:
            base_dir = output_base_path
            base_name = DEFAULT_OUTPUT_NAME

        return base_dir, base_name

    @staticmethod
    def ensure_output_directories(paths: Dict[str, str]) -> None:
        os.makedirs(paths["output_dir"], exist_ok=True)
        if paths["match_dir"]:
            os.makedirs(paths["match_dir"], exist_ok=True)

    @staticmethod
    def save_final_image(final_image_path: str, final_image: "np.ndarray") -> None:
        cv2.imwrite(final_image_path, final_image)

    @staticmethod
    def save_match_visualization(match_dir: str, vis_img: "np.ndarray",
                                 img1_name: str, img2_name: str, idx1: int, idx2: int) -> None:
        if not match_dir:
            return

        save_path = os.path.join(match_dir, get_match_pattern(idx1, idx2, img1_name, img2_name))
        cv2.imwrite(save_path, vis_img)


class StitchingInfoManager:
    @staticmethod
    def save_stitching_info(paths: Dict[str, str], image_folder: str, save_matches: bool,
                            custom_order: Dict, sorted_names: List[str],
                            sorted_images: List["np.ndarray"], crop_info: List[Tuple[int, int]],
                            total_original_height: int, final_height: int,
                            height_saved: int, saving_ratio: float) -> None:
        stitching_info = {
            "image_folder": image_folder,
            "final_image_path": paths["final_image_path"],
            "output_directory": paths["output_dir"],
            "image_count": len(sorted_names),
            "save_matches": save_matches,
            "match_directory": paths["match_dir"],
            "custom_order": custom_order,
            "original_height": total_original_height,
            "final_height": final_height,
            "height_saved": height_saved,
            "saving_ratio": saving_ratio,
            "image_order": sorted_names,
            "crop_info": StitchingInfoManager._create_crop_info(
                sorted_names, sorted_images, crop_info),
            "output_files": {
                "final_image": paths["final_image_path"],
                "info_file": paths["info_file"],
                "match_directory": paths["match_dir"] if save_matches else None
            }
        }

        with open(paths["info_file"], 'w', encoding='utf-8') as f:
            json.dump(stitching_info, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _create_crop_info(sorted_names: List[str], sorted_images: List["np.ndarray"],
                          crop_info: List[Tuple[int, int]]) -> List[Dict]:
        return [
            {
                "filename": sorted_names[i],
                "original_height": sorted_images[i].shape[0],
                "crop_top": crop_info[i][0],
                "crop_bottom": crop_info[i][1],
                "final_height": crop_info[i][1] - crop_info[i][0],
                "saved_height": sorted_images[i].shape[0] - (crop_info[i][1] - crop_info[i][0])
            }
            for i in range(len(sorted_names))
        ]
