import cv2
import numpy as np
import os
import shutil
from typing import List, Tuple, Dict

class AKAZEMatcher:
    def __init__(self):
        self.akaze = cv2.AKAZE_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def load_images(self, folder: str, idx1: int, idx2: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        image_files = sorted([f for f in os.listdir(folder)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

        if idx1 < 1 or idx2 < 1 or idx1 > len(image_files) or idx2 > len(image_files):
            print(f"[错误] 图像索引无效！可用范围: 1-{len(image_files)}")
            return None, None, None, None

        img1_path = os.path.join(folder, image_files[idx1 - 1])
        img2_path = os.path.join(folder, image_files[idx2 - 1])

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print("[错误] 无法加载图像")
            return None, None, None, None

        print(f"[信息] 加载图像:")
        print(f"  图像{idx1}: {os.path.basename(img1_path)} - {img1.shape[:2]}")
        print(f"  图像{idx2}: {os.path.basename(img2_path)} - {img2.shape[:2]}")

        return img1, img2, os.path.basename(img1_path), os.path.basename(img2_path)

    def detect_akaze_features(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        keypoints, descriptors = self.akaze.detectAndCompute(gray, None)

        print(f"[特征] AKAZE特征点检测: {len(keypoints)} 个关键点")
        if descriptors is not None:
            print(f"[特征] 描述符维度: {descriptors.shape}")

        return keypoints, descriptors

    def analyze_match_regions(self, kp1: List, kp2: List, matches: List,
                            img1_shape: Tuple, img2_shape: Tuple) -> Tuple[int, float, float]:
        if len(matches) < 5:
            return 0, 0.0, 0.0

        h1, w1 = img1_shape[:2]
        h2, w2 = img2_shape[:2]

        mid_y1 = h1 / 2
        mid_y2 = h2 / 2

        img1_upper_count = 0
        img1_lower_count = 0
        img2_upper_count = 0
        img2_lower_count = 0

        img1_y_positions = []
        img2_y_positions = []

        for m in matches:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt

            img1_y_positions.append(pt1[1])
            img2_y_positions.append(pt2[1])

            if pt1[1] < mid_y1:
                img1_upper_count += 1
            else:
                img1_lower_count += 1

            if pt2[1] < mid_y2:
                img2_upper_count += 1
            else:
                img2_lower_count += 1

        avg_y1 = np.mean(img1_y_positions)
        avg_y2 = np.mean(img2_y_positions)

        print(f"\n[分析] 区域分析:")
        print(f"  图像1: 上半区 {img1_upper_count} 点, 下半区 {img1_lower_count} 点")
        print(f"  图像2: 上半区 {img2_upper_count} 点, 下半区 {img2_lower_count} 点")
        print(f"  图像1平均Y位置: {avg_y1:.1f} (总高度: {h1})")
        print(f"  图像2平均Y位置: {avg_y2:.1f} (总高度: {h2})")

        if (img1_lower_count > img1_upper_count and
            img2_upper_count > img2_lower_count and
            len(matches) >= 8):
            return -1, avg_y1, avg_y2
        elif (img1_upper_count > img1_lower_count and
              img2_lower_count > img2_upper_count and
              len(matches) >= 8):
            return 1, avg_y1, avg_y2
        else:
            if avg_y1 > mid_y1 and avg_y2 < mid_y2:
                return -1, avg_y1, avg_y2
            elif avg_y1 < mid_y1 and avg_y2 > mid_y2:
                return 1, avg_y1, avg_y2
            else:
                return 0, avg_y1, avg_y2

    def match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[List, np.ndarray, int]:
        print("\n=== AKAZE特征匹配 ===")

        kp1, desc1 = self.detect_akaze_features(img1)
        kp2, desc2 = self.detect_akaze_features(img2)

        if desc1 is None or desc2 is None:
            print("[错误] 无法获取特征描述符")
            return [], None, 0

        print("[匹配] 正在进行特征匹配...")
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        print(f"[匹配] 初始匹配数: {len(matches)}")

        if len(matches) > 10:
            good_matches = matches[:len(matches)//2]
        else:
            good_matches = matches[:len(matches)//3] if len(matches) > 3 else matches

        print(f"[匹配] 距离筛选后: {len(good_matches)} 个匹配")

        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            if homography is not None:
                inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask.ravel()[i] == 1]
                print(f"[验证] 几何验证后: {len(inlier_matches)} 个内点")

                order, avg_y1, avg_y2 = self.analyze_match_regions(
                    kp1, kp2, inlier_matches, img1.shape, img2.shape)

                inlier_ratio = len(inlier_matches) / len(good_matches) * 100
                print(f"[质量] 内点比例: {inlier_ratio:.1f}%")

                if order == -1:
                    print(f"[分析] 拼接顺序: 图像1 -> 图像2 (图像1在上方)")
                elif order == 1:
                    print(f"[分析] 拼接顺序: 图像2 -> 图像1 (图像2在上方)")
                else:
                    print(f"[分析] 拼接顺序: 无法确定")

                return inlier_matches, homography, order
            else:
                print("[错误] 无法计算有效的单应性矩阵")
        else:
            print("[错误] 匹配点数量不足，无法进行几何验证")

        return [], None, 0

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

        cv2.putText(info_bar, f"AKAZE匹配 - {title}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_bar, f"匹配点: {len(matches)}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(info_bar, f"特征点: {len(kp1)} / {len(kp2)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        result = np.vstack([info_bar, match_image])

        if w > 1600:
            scale = 1600 / w
            result = cv2.resize(result, (1600, int((h + info_height) * scale)))

        return result

class ImageSorter:
    def __init__(self, image_folder: str):
        self.image_folder = image_folder
        self.matcher = AKAZEMatcher()
        self.images = []
        self.image_names = []
        self.original_paths = []
        self.load_images()

    def load_images(self):
        print(f"正在加载图像文件夹: {self.image_folder}")

        if not os.path.exists(self.image_folder):
            print(f"[错误] 图像文件夹 '{self.image_folder}' 不存在")
            return

        image_files = sorted([f for f in os.listdir(self.image_folder)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

        if len(image_files) < 2:
            print(f"[错误] 图像数量不足，需要至少2张图像，当前: {len(image_files)}")
            return

        print(f"[信息] 找到 {len(image_files)} 张图像:")

        for i, filename in enumerate(image_files):
            img_path = os.path.join(self.image_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[警告] 无法加载: {filename}")
                continue

            self.images.append(img)
            self.image_names.append(filename)
            self.original_paths.append(img_path)
            print(f"  {i+1}. {filename} - {img.shape[:2]}")

        print(f"[成功] 成功加载 {len(self.images)} 张图像")

    def analyze_all_pairs(self) -> Dict[Tuple[int, int], Dict]:
        if len(self.images) < 2:
            return {}

        print(f"\n{'='*80}")
        print("开始分析所有图像对的匹配关系...")
        print(f"{'='*80}")

        results = {}
        n = len(self.images)

        for i in range(n):
            for j in range(n):
                if i != j:
                    print(f"\n--- 分析图像{i+1} vs 图像{j+1} ---")

                    img1, img2, name1, name2 = self.matcher.load_images(
                        self.image_folder, i+1, j+1)

                    if img1 is not None and img2 is not None:
                        matches, homography, order = self.matcher.match_features(img1, img2)

                        if matches and homography is not None and order != 0:
                            results[(i, j)] = {
                                'img1_idx': i,
                                'img2_idx': j,
                                'order': order,
                                'pair_name': f"{name1} <-> {name2}",
                                'img1_shape': img1.shape,
                                'img2_shape': img2.shape,
                                'matches_count': len(matches)
                            }
                            print(f"[成功] 分析结果: {name1} <-> {name2} - 顺序: {order}")

        print(f"\n{'='*80}")
        print(f"分析完成，共找到 {len(results)} 组有效的匹配关系")
        print(f"{'='*80}")

        return results

    def find_optimal_order(self, match_results: Dict[Tuple[int, int], Dict]) -> List[int]:
        if len(match_results) == 0:
            print("[错误] 没有找到有效的匹配关系")
            return []

        print(f"\n[分析] 开始寻找最优拼接顺序...")

        image_scores = {}

        for (i, j), result in match_results.items():
            if i not in image_scores:
                image_scores[i] = [0, 0]
            if j not in image_scores:
                image_scores[j] = [0, 0]

            if result['order'] == -1:
                image_scores[i][0] += 1
                image_scores[j][1] += 1
            elif result['order'] == 1:
                image_scores[i][1] += 1
                image_scores[j][0] += 1

        best_top = None
        best_score = -float('inf')

        for idx, (up_score, down_score) in image_scores.items():
            combined_score = up_score - down_score
            print(f"  图像{idx+1}: 上方={up_score}, 下方={down_score}, 综合={combined_score}")

            if combined_score > best_score:
                best_score = combined_score
                best_top = idx

        if best_top is None:
            print("[错误] 无法确定起始图像")
            return []

        print(f"[成功] 确定起始图像: 图像{best_top+1}")

        ordered_images = [best_top]
        used_images = {best_top}

        for _ in range(len(self.images) - 1):
            current = ordered_images[-1]
            best_next = None
            best_confidence = 0

            print(f"\n[分析] 寻找图像{current+1}的下一张图像...")

            for (i, j), result in match_results.items():
                if i == current and j not in used_images and result['order'] == -1:
                    confidence = result['matches_count']
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_next = j
                elif j == current and i not in used_images and result['order'] == 1:
                    confidence = result['matches_count']
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_next = i

            if best_next is not None:
                ordered_images.append(best_next)
                used_images.add(best_next)
                print(f"[成功] 选择图像{best_next+1}作为下一个图像")
            else:
                remaining = [idx for idx in range(len(self.images)) if idx not in used_images]
                if remaining:
                    ordered_images.append(remaining[0])
                    used_images.add(remaining[0])
                    print(f"[警告] 无法确定明确顺序，随机选择图像{remaining[0]+1}")

        print(f"\n[结果] 最终拼接顺序: {[idx+1 for idx in ordered_images]}")
        return ordered_images

    def save_order_result(self, image_order: List[int]):
        output_dir = "matches"
        os.makedirs(output_dir, exist_ok=True)

        order_file = os.path.join(output_dir, "stitching_order.txt")
        with open(order_file, 'w', encoding='utf-8') as f:
            f.write("图像拼接顺序:\n")
            f.write("-" * 40 + "\n")
            for i, idx in enumerate(image_order):
                f.write(f"{i+1:02d}. {self.image_names[idx]} (原始索引: {idx+1})\n")
            f.write("-" * 40 + "\n")
            f.write(f"排序后的索引序列: {[idx+1 for idx in image_order]}\n")
            f.write(f"排序后的文件列表: {[self.image_names[idx] for idx in image_order]}\n")

        print(f"[信息] 拼接顺序已保存: {order_file}")

        mapping_file = os.path.join(output_dir, "index_mapping.txt")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            f.write("索引映射关系:\n")
            f.write("-" * 40 + "\n")
            f.write("原始索引 -> 排序后位置\n")
            for original_idx, sorted_idx in enumerate(image_order):
                f.write(f"{original_idx+1} -> {sorted_idx+1} ({self.image_names[original_idx]})\n")

        print(f"[信息] 索引映射已保存: {mapping_file}")

    def rename_images(self, image_order: List[int]):
        print(f"\n{'='*80}")
        print("开始按正确顺序重命名图像...")
        print(f"{'='*80}")

        sorted_dir = "../opencv/sorted_images"
        os.makedirs(sorted_dir, exist_ok=True)

        order_file = os.path.join(sorted_dir, "order_info.txt")
        with open(order_file, 'w', encoding='utf-8') as f:
            f.write("图像重命名顺序:\n")
            f.write("-" * 40 + "\n")

        for new_idx, original_idx in enumerate(image_order):
            original_path = self.original_paths[original_idx]
            original_name = self.image_names[original_idx]
            new_name = f"{new_idx+1:02d}_{original_name}"
            new_path = os.path.join(sorted_dir, new_name)

            print(f"  {new_idx+1:02d}. {original_name} -> {new_name}")

            shutil.copy2(original_path, new_path)

            with open(order_file, 'a', encoding='utf-8') as f:
                f.write(f"{new_idx+1:02d}. {original_name} (来自原始索引: {original_idx+1})\n")

        print(f"\n{'='*80}")
        print(f"图像重命名完成！")
        print(f"{'='*80}")
        print(f"[信息] 排序后的图像保存在: {sorted_dir}")
        print(f"[信息] 排序信息保存在: {order_file}")

    def generate_adjacent_visualizations(self, image_order: List[int]):
        print(f"\n{'='*80}")
        print("生成相邻图像对的匹配可视化...")
        print(f"{'='*80}")

        output_dir = "matches"
        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(image_order) - 1):
            idx1 = image_order[i]
            idx2 = image_order[i + 1]

            print(f"\n--- 生成相邻图像对 {idx1+1} vs {idx2+1} 的可视化 ---")

            img1, img2, name1, name2 = self.matcher.load_images(
                self.image_folder, idx1 + 1, idx2 + 1)

            if img1 is not None and img2 is not None:
                matches, homography, order = self.matcher.match_features(img1, img2)

                if matches and homography is not None:
                    kp1, _ = self.matcher.detect_akaze_features(img1)
                    kp2, _ = self.matcher.detect_akaze_features(img2)

                    vis_img = self.matcher.visualize_matches(
                        img1, img2, kp1, kp2, matches,
                        f"{name1} vs {name2}"
                    )

                    save_path = os.path.join(output_dir, f"match_{i+1:02d}_{i+2:02d}.jpg")
                    cv2.imwrite(save_path, vis_img)
                    print(f"[保存] 匹配结果已保存: {save_path}")

                    if len(matches) >= 10:
                        print(f"[成功] 匹配质量: 良好 ({len(matches)} 个匹配点)")
                    elif len(matches) >= 4:
                        print(f"[警告] 匹配质量: 一般 ({len(matches)} 个匹配点)")
                    else:
                        print(f"[失败] 匹配质量: 较差 ({len(matches)} 个匹配点)")
                else:
                    print(f"[失败] 无法生成相邻对可视化: {name1} <-> {name2}")
            else:
                print(f"[失败] 无法加载图像: {name1} <-> {name2}")

        print(f"\n{'='*80}")
        print("相邻图像匹配可视化生成完成！")
        print(f"{'='*80}")

    def process_all(self):
        print(f"[开始] 开始图像序列分析流程...")

        if len(self.images) < 2:
            print("[错误] 图像数量不足，无法进行分析")
            return None

        match_results = self.analyze_all_pairs()
        if not match_results:
            print("[错误] 没有找到有效的匹配关系")
            return None

        image_order = self.find_optimal_order(match_results)
        if len(image_order) < 2:
            print("[错误] 无法确定有效的拼接顺序")
            return None

        self.save_order_result(image_order)
        self.generate_adjacent_visualizations(image_order)
        self.rename_images(image_order)

        print(f"\n[完成] 图像排序完成！")
        return image_order

def main():
    print(f"图像排序和重命名工具")
    print(f"{'='*50}")

    image_folder = input("请输入图像文件夹路径: ").strip().strip('"\'')

    if not image_folder:
        print("[错误] 图像文件夹路径不能为空")
        return

    image_folder = os.path.normpath(image_folder)

    print(f"使用图像文件夹: {image_folder}")

    sorter = ImageSorter(image_folder)
    image_order = sorter.process_all()

    if image_order is not None:
        print(f"\n[成功] 排序完成！")
        print(f"[下一步] 请运行 02_stitch_images.py 对 {image_folder}/sorted_images 目录进行拼接")
    else:
        print(f"\n[失败] 排序失败，请检查图像质量和重叠区域。")

if __name__ == "__main__":
    main()