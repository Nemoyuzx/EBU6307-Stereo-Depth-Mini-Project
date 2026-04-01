from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage


from common import discover_scenes, filter_scene_dirs, write_scene_text
from config import O2Config

# 决定 CSV 的写出顺序，旧指标文件在回读时按什么字段补齐。
METRIC_FIELDNAMES = [
    "scene",
    "transform_family",
    "random_seed",
    "left_keypoints",
    "right_keypoints",
    "raw_knn_matches",
    "ratio_test_matches",
    "repeatable_matches",
    "repeatability",
    "repeatability_proxy",
    "homography_threshold_px",
    "transform_params_json",
]

DISTINCT_TRANSFORM_FAMILIES = (
    "rotation",
    "affine",
    "intensity",
)

MAX_VISUALIZED_SIFT_ITEMS = 20

MetricValue = str | float | int
MetricRow = dict[str, MetricValue]
TransformParams = dict[str, MetricValue]


@dataclass(frozen=True)
class ManualKeypoint:
    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int
    layer: int
    sigma: float

    @property
    def pt(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class ManualMatch:
    queryIdx: int
    trainIdx: int
    distance: float


@dataclass(frozen=True)
class _ScaleSpaceCandidate:
    x: float
    y: float
    octave: int
    layer: int
    sigma: float
    response: float


class ManualSiftDetector:
    def __init__(self, max_features: int, contrast_threshold: float) -> None:
        self.max_features = max(1, int(max_features))  # 最多保留多少个关键点，避免后续描述与匹配数量失控。
        self.contrast_threshold = max(float(contrast_threshold), 1e-4)  # DoG 极值的最小对比度阈值，越大越严格。
        self.scales_per_octave = 3  # 每个 octave 细分成多少个尺度层，用来构建高斯/DoG 金字塔。
        self.base_sigma = 1.6  # SIFT 基础高斯模糊标准差，决定金字塔第一层的平滑强度。
        self.max_octaves = 4  # 最多构建多少个 octave，控制跨尺度检测范围。
        self.edge_threshold = 10.0  # 边缘响应剔除阈值，用 Hessian 主曲率比过滤细长边缘点。
        self.orientation_magnitude_ratio = 0.1  # 方向分配时丢弃过弱梯度，降低噪声方向对直方图的扰动。
        self.orientation_window_size = 16  # 方向分配固定使用16x16窗口，对齐课件里的方向直方图统计区域。
        self.orientation_block_size = 2  # 方向分配先把16x16窗口切成2x2小块，再为每个小块计算一个聚合梯度方向。
        self.descriptor_window_size = 16.0  # 描述子在主方向和尺度归一化后固定映射到16x16窗口。
        self.descriptor_block_count = 4  # 16x16窗口切成4x4个子区域。
        self.descriptor_orientation_bins = 8  # 每个4x4子区域统计8个方向bin。
        self.descriptor_clip_value = 0.2  # 第一次归一化后把过大的bin截断到0.2，抑制局部强梯度主导描述子。

    def detectAndCompute(self, image: Any, mask: Any) -> tuple[list[ManualKeypoint], np.ndarray | None]:

        # mask参数丢弃
        if mask is not None:
            raise NotImplementedError("ManualSiftDetector does not support masks.")

        # 输入检查与预处理：确保输入图像是单通道灰度图，并且像素值在 0~1 范围内
        gray = np.asarray(image, dtype=np.float32)
        if gray.ndim != 2:
            raise ValueError("ManualSiftDetector expects a 2D grayscale image.")
        if gray.size == 0:
            return [], None
        if float(gray.max(initial=0.0)) > 1.0:
            gray /= 255.0

        # 第一步：构建尺度空间，检测 DoG 极值并精确定位，得到初始候选关键点列表。
        # 构建金字塔
        gaussian_pyramid, dog_pyramid, sigma_pyramid = self._build_scale_space(gray) # Scale-Space Extrema Detection
        
        # Scale-Space Extrema Detection & Key Point Localization
        # 第二步：对每个候选点，提取其所在位置的梯度信息，计算主方向，并构建描述子。最终输出 keypoints 和 descriptors。
        candidates = self._detect_candidates(dog_pyramid, sigma_pyramid) # DoG 尺度空间里，把所有可能成为 SIFT 关键点的位置先找出来，产出一批候选点。这里的 candidates 还不是最终输出给匹配的 keypoints，而是中间结果。 
        if not candidates:
            return [], None

        # 降低候选点之间的空间重复度，避免过多响应几乎重合的点占满 max_features 的配额，提升结果的分布和多样性。课件上没有，opencv的操作是基于距离的非极大值抑制，效果不错，这里为了简单直接就改成了直接比较坐标距离的抑制，感觉差不多。
        selected_candidates = self._suppress_candidates(candidates) 
        keypoints: list[ManualKeypoint] = []
        descriptors: list[np.ndarray] = []

        # 第三步：方向分配与描述子构建。对于每个候选点，提取其所在位置的梯度信息，计算主方向，并构建描述子。最终输出 keypoints 和 descriptors。
        gradient_pyramid = self._build_gradient_pyramid(gaussian_pyramid) # 提取梯度信息
        for candidate in selected_candidates: # 对每个候选点
            magnitude, orientation = gradient_pyramid[candidate.octave][candidate.layer] # 提取其所在位置的梯度信息
            angles = self._assign_orientations(magnitude, orientation, candidate.x, candidate.y, candidate.sigma) # Orientation Assignment
            if not angles: # 如果这个候选点的主方向提取失败了（可能因为梯度信息不足或不稳定），就放弃这个点，不输出对应的 keypoint 和 descriptor。
                continue
            for angle in angles:
                descriptor = self._build_descriptor(magnitude, orientation, candidate.x, candidate.y, candidate.sigma, angle) # Descriptor Construction
                if descriptor is None: # 可能因为描述子窗口越界或梯度信息不足导致无法构建有效描述子，这时就跳过这个候选点。
                    continue

                scale_factor = float(2 ** candidate.octave) # 根据 octave 计算尺度因子，将候选点坐标从 octave 空间映射回原始图像空间
                keypoints.append(
                    ManualKeypoint(
                        x=float(candidate.x * scale_factor),
                        y=float(candidate.y * scale_factor),
                        size=float(candidate.sigma * scale_factor * 2.0),
                        angle=float(angle),
                        response=float(candidate.response),
                        octave=candidate.octave,
                        layer=candidate.layer,
                        sigma=float(candidate.sigma * scale_factor),
                    )
                )
                descriptors.append(descriptor)
                if len(keypoints) >= self.max_features:
                    break
            if len(keypoints) >= self.max_features:
                break

        if not descriptors:
            return [], None
        return keypoints, np.asarray(descriptors, dtype=np.float32)

    def _build_scale_space(self, image: np.ndarray) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[list[float]]]:
        '''第一步：尺度空间构建与极值检测。
        标准 SIFT 先构造高斯尺度空间 L(x,y,sigma)=G(x,y,sigma)*I(x,y)，其中 G(x,y,sigma)=1/(2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2))。
        sigma 越大，图像越模糊，也就代表越粗的观察尺度。
        每个 octave 内按 k=2^(1/s) 递增 sigma，得到同一组里不同模糊强度的图像。
        
        gaussian_pyramid: list[list[np.ndarray]] = [] # 高斯金字塔
        dog_pyramid: list[list[np.ndarray]] = [] # 高斯差分DoG金字塔
        sigma_pyramid: list[list[float]] = [] # sigma 金字塔'''
        # 第一步：尺度空间极值检测。
        # 标准 SIFT 先构造高斯尺度空间 L(x,y,sigma)=G(x,y,sigma)*I(x,y)，其中
        # G(x,y,sigma)=1/(2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2))。
        # sigma 越大，图像越模糊，也就代表越粗的观察尺度。
        gaussian_pyramid: list[list[np.ndarray]] = [] # 高斯金字塔
        dog_pyramid: list[list[np.ndarray]] = [] # 高斯差分DoG金字塔
        sigma_pyramid: list[list[float]] = [] # sigma 金字塔

        current = ndimage.gaussian_filter(image, sigma=self.base_sigma, mode="reflect").astype(np.float32)
        min_dimension = min(image.shape[:2])
        dynamic_octaves = max(1, int(math.floor(math.log2(max(min_dimension, 16)))) - 4) # 根据图像尺寸动态决定 octave 数量，确保最小层的有效观察范围不小于 16x16
        octave_count = max(1, min(self.max_octaves, dynamic_octaves)) # 实际构建的 octave 数量不能超过 max_octaves，同时也不能超过图像尺寸允许的范围 （/rough）
        k = 2.0 ** (1.0 / self.scales_per_octave) # 每个 octave 内 sigma 的递增因子 （k = 2^(1/{/rough})）
        # 每个 octave 内的层数：scales_per_octave 个正常层 + 1 个额外模糊层（用于 DoG 计算） + 1 个额外模糊层（用于下一 octave 的 base image）,
        # 为了覆盖完整的倍频程，这意味着需要在倍频程图像之后额外增加两幅图像，从而使图像总数达到 𝜌 + 3 幅。(课件)
        levels_per_octave = self.scales_per_octave + 3 
        
        # 每组会生成  levels_per_octave （{/rough+3}） 张高斯模糊图像，相减得到 levels_per_octave - 1（{/rough+2}） 张 DoG 图像，最终能在中间的 scales_per_octave （{/rough}） 层 DoG 图像上寻找极值点。

        for octave in range(octave_count):
            octave_gaussians = [current]
            octave_sigmas = [self.base_sigma]
            for layer in range(1, levels_per_octave):
                # 每个 octave 内按 k=2^(1/s) 递增 sigma，得到同一组里不同模糊强度的图像。
                prev_sigma = self.base_sigma * (k ** (layer - 1))
                total_sigma = self.base_sigma * (k ** layer)
                incremental_sigma = math.sqrt(max(total_sigma * total_sigma - prev_sigma * prev_sigma, 1e-6)) # 增量模糊强度，确保每层都是在前一层基础上增加模糊，而不是直接从原图模糊到目标 sigma，避免重复模糊导致信息损失。
                octave_gaussians.append(
                    ndimage.gaussian_filter(octave_gaussians[-1], sigma=incremental_sigma, mode="reflect").astype(np.float32)
                )
                octave_sigmas.append(total_sigma)

            gaussian_pyramid.append(octave_gaussians)
            sigma_pyramid.append(octave_sigmas)
            dog_pyramid.append(
                # 高斯差分 DoG: D(x,y,sigma)=L(x,y,k*sigma)-L(x,y,sigma)，它是 LoG 的高效近似。
                [octave_gaussians[layer + 1] - octave_gaussians[layer] for layer in range(len(octave_gaussians) - 1)]
            )

            # 一个 octave 结束后，把中间层降采样为下一组的 base image，形成金字塔结构。
            # 这对应标准 SIFT 中“第一组看原始分辨率，下一组看尺寸减半图像”的做法。
            next_base = octave_gaussians[self.scales_per_octave][::2, ::2] #the first image of a new octave can be obtained directly by downsampling that third image of the previous octave by 2. 新八度的第一幅图像可以直接通过对上一八度的第三幅图像进行 2 倍下采样获得。
            if min(next_base.shape[:2]) < 24: # 如果下一 octave 的 base image 太小了，就没什么意义继续构建更小的 octave 了，退出构建
                break
            current = next_base.astype(np.float32)

        return gaussian_pyramid, dog_pyramid, sigma_pyramid

    def _build_gradient_pyramid(self, gaussian_pyramid: list[list[np.ndarray]]) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        '''梯度信息计算 邻域内所有像素的梯度幅值 $m(x,y)$ 和梯度方向 $theta(x,y)$'''
        # 第三步和第四步都会复用梯度信息，所以先在每个尺度图像上预计算：
        # 邻域内所有像素的梯度幅值 $m(x,y)$ 和梯度方向 $\theta(x,y)$
        # m(x,y)=sqrt((L(x+1,y)-L(x-1,y))^2 + (L(x,y+1)-L(x,y-1))^2)
        # theta(x,y)=atan2(L(x,y+1)-L(x,y-1), L(x+1,y)-L(x-1,y))。
        gradient_pyramid: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for octave_images in gaussian_pyramid:
            octave_gradients: list[tuple[np.ndarray, np.ndarray]] = []
            for image in octave_images:
                grad_x = ndimage.sobel(image, axis=1, mode="reflect")
                grad_y = ndimage.sobel(image, axis=0, mode="reflect")
                magnitude = np.hypot(grad_x, grad_y).astype(np.float32)
                orientation = (np.degrees(np.arctan2(grad_y, grad_x)) + 360.0) % 360.0
                octave_gradients.append((magnitude, orientation.astype(np.float32)))
            gradient_pyramid.append(octave_gradients)
        return gradient_pyramid

    def _detect_candidates( self, dog_pyramid: list[list[np.ndarray]] , sigma_pyramid: list[list[float]],) -> list[_ScaleSpaceCandidate]:
        '''
        第一步的 DoG 极值搜索：每个点都要和 26 个邻域点比较，分别是
        同尺度 8 个邻点、上一尺度 9 个邻点、下一尺度 9 个邻点。
        只有当它是这 26 个点里的最大值或最小值时，才会成为初始候选关键点。
        
        第二步：对初始极值做 3D 泰勒展开近似，求连续空间中的
        X_hat=(delta_x, delta_y, delta_sigma)，把像素级候选点细化到子像素/子尺度位置。
        当某个维度上的偏移超过 0.5 时，说明更接近相邻采样点，需要迭代更新中心再重算。
        '''
        candidates: list[_ScaleSpaceCandidate] = []
        contrast_floor = max(0.001, self.contrast_threshold / float(self.scales_per_octave)) 
        # 点的D_extremum值必须大于 contrast_floor，才能被认为是有效的候选点。“weak extrema” or “low contrast points”
        # 因为这里筛的是 DoG 的响应值，而 DoG 的数值大小本身会随着每个 octave 里分多少层而变。（本质是在做一个近似归一化，让不同尺度层密度下的 DoG 响应阈值保持大致可比，同opencv）

        for octave, octave_dogs in enumerate(dog_pyramid):
            for layer in range(1, len(octave_dogs) - 1):
                #第一步的 DoG 极值搜索：每个点都要和 26 个邻域点比较，分别是同尺度 8 个邻点、上一尺度 9 个邻点、下一尺度 9
                previous = octave_dogs[layer - 1]
                current = octave_dogs[layer]
                following = octave_dogs[layer + 1]
                stacked = np.stack([previous, current, following], axis=0)
                local_max = ndimage.maximum_filter(stacked, size=(3, 3, 3), mode="nearest")[1]
                local_min = ndimage.minimum_filter(stacked, size=(3, 3, 3), mode="nearest")[1]
                candidate_mask = (
                    ((current >= local_max) | (current <= local_min))
                    & (np.abs(current) >= contrast_floor) # 如果该点的值大于其所有邻域像素的值，或小于其所有邻域像素的值，同时大于 contrast_floor，则该点被选为极值点
                )

                # 边界点没有完整的 3x3x3 邻域，既不利于极值比较，也不利于后续描述子提取。
                border = 8
                candidate_mask[:border, :] = False
                candidate_mask[-border:, :] = False
                candidate_mask[:, :border] = False
                candidate_mask[:, -border:] = False

                
                # 第二步：对初始极值做 3D 泰勒展开近似，求连续空间中的 X_hat=(delta_x, delta_y, delta_sigma)，把像素级候选点细化到子像素/子尺度位置。
                ys, xs = np.nonzero(candidate_mask) # 找到候选点的坐标
                for y, x in zip(ys.tolist(), xs.tolist()):
                    candidate = self._refine_candidate_location( # 对每个初始候选点，做 3D 二阶泰勒展开，求连续空间中的偏移，把像素级候选点细化到子像素/子尺度位置。
                        octave=octave,
                        octave_dogs=octave_dogs,
                        octave_sigmas=sigma_pyramid[octave],
                        layer=layer,
                        y=int(y),
                        x=int(x),
                        contrast_floor=contrast_floor,
                    )
                    if candidate is None:
                        continue
                    edge_y = int(round(candidate.y))
                    edge_x = int(round(candidate.x))
                    if not self._passes_edge_response(octave_dogs[candidate.layer], edge_y, edge_x): # 对细化后的候选点进行边缘响应剔除，如果该点的主曲率比率超过阈值，说明该点更像边缘响应而不是稳定角点，需要剔除。
                        continue
                    candidates.append(candidate)

        candidates.sort(key=lambda candidate: candidate.response, reverse=True)
        return candidates

    def _refine_candidate_location(self, octave: int, octave_dogs: list[np.ndarray], octave_sigmas: list[float], layer: int, y: int, x: int, contrast_floor: float,) -> _ScaleSpaceCandidate | None:
        '''步骤 2：关键点定位'''
        # 第二步：关键点精确定位。
        # 对 D(x,y,sigma) 在当前采样点附近做二阶泰勒展开：
        # D(X)=D + dD/dX^T X + 1/2 * X^T * d2D/dX2 * X。
        # 令导数为 0 后可得到偏移 X_hat=-H^-1 g，其中 g 是一阶导，H 是 3D Hessian。
        # 这样可以把离散极值细化到连续位置，并顺带更准确地评估对比度。
        max_iterations = 5
        offset = np.zeros(3, dtype=np.float32)

        for _ in range(max_iterations):
            if layer <= 0 or layer >= len(octave_dogs) - 1:
                return None
            current = octave_dogs[layer]
            height, width = current.shape
            if x <= 0 or x >= width - 1 or y <= 0 or y >= height - 1:
                return None

            gradient = self._dog_gradient(octave_dogs, layer, y, x)
            hessian = self._dog_hessian(octave_dogs, layer, y, x)
            try:
                offset = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                return None

            if np.all(np.abs(offset) < 0.5):
                break

            x += int(math.copysign(1.0, float(offset[0]))) if abs(float(offset[0])) >= 0.5 else 0
            y += int(math.copysign(1.0, float(offset[1]))) if abs(float(offset[1])) >= 0.5 else 0
            layer += int(math.copysign(1.0, float(offset[2]))) if abs(float(offset[2])) >= 0.5 else 0
        else:
            return None

        current = octave_dogs[layer]
        interpolated_response = float(current[y, x] + 0.5 * gradient.dot(offset))
        if abs(interpolated_response) < contrast_floor:
            return None

        refined_x = float(x + offset[0])
        refined_y = float(y + offset[1])
        if refined_x < 8.0 or refined_x >= current.shape[1] - 8.0:
            return None
        if refined_y < 8.0 or refined_y >= current.shape[0] - 8.0:
            return None

        refined_sigma = float(octave_sigmas[layer] * (2.0 ** (float(offset[2]) / self.scales_per_octave)))
        return _ScaleSpaceCandidate(
            x=refined_x,
            y=refined_y,
            octave=octave,
            layer=layer,
            sigma=refined_sigma,
            response=abs(interpolated_response),
        )

    def _dog_gradient(self, octave_dogs: list[np.ndarray], layer: int, y: int, x: int) -> np.ndarray:
        '''二阶泰勒展开的一阶导数 dD/dX，中心差分，分别对 x、y、尺度 s 三个维度求导'''
        previous = octave_dogs[layer - 1]
        current = octave_dogs[layer]
        following = octave_dogs[layer + 1]
        return np.asarray(
            [
                (current[y, x + 1] - current[y, x - 1]) * 0.5,
                (current[y + 1, x] - current[y - 1, x]) * 0.5,
                (following[y, x] - previous[y, x]) * 0.5,
            ],
            dtype=np.float32,
        )

    def _dog_hessian(self, octave_dogs: list[np.ndarray], layer: int, y: int, x: int) -> np.ndarray:
        '''二阶泰勒展开的二阶导数 d2D/dX2，中心差分，分别对 x、y、尺度 s 三个维度求导'''
        previous = octave_dogs[layer - 1]
        current = octave_dogs[layer]
        following = octave_dogs[layer + 1]
        dxx = current[y, x + 1] + current[y, x - 1] - 2.0 * current[y, x]
        dyy = current[y + 1, x] + current[y - 1, x] - 2.0 * current[y, x]
        dss = following[y, x] + previous[y, x] - 2.0 * current[y, x]
        dxy = (current[y + 1, x + 1] - current[y + 1, x - 1] - current[y - 1, x + 1] + current[y - 1, x - 1]) * 0.25
        dxs = (following[y, x + 1] - following[y, x - 1] - previous[y, x + 1] + previous[y, x - 1]) * 0.25
        dys = (following[y + 1, x] - following[y - 1, x] - previous[y + 1, x] + previous[y - 1, x]) * 0.25
        return np.asarray(
            [
                [dxx, dxy, dxs],
                [dxy, dyy, dys],
                [dxs, dys, dss],
            ],
            dtype=np.float32,
        )

    def _passes_edge_response(self, dog_image: np.ndarray, y: int, x: int) -> bool:
        """ 标准 SIFT 使用 Hessian 矩阵 H=[[Dxx,Dxy],[Dxy,Dyy]] 估计主曲率。
        若 Tr(H)^2 / Det(H) >= ((r+1)^2) / r，则说明一个方向曲率远大于另一个方向，
        该点更像边缘上的响应而不是稳定角点，需要剔除。这里的 r 就是 edge_threshold。 """
        dxx = float(dog_image[y, x + 1] + dog_image[y, x - 1] - 2.0 * dog_image[y, x])
        dyy = float(dog_image[y + 1, x] + dog_image[y - 1, x] - 2.0 * dog_image[y, x])
        dxy = float(
            dog_image[y + 1, x + 1]
            - dog_image[y + 1, x - 1]
            - dog_image[y - 1, x + 1]
            + dog_image[y - 1, x - 1]
        ) / 4.0
        trace = dxx + dyy # Tr(H) = Dxx + Dyy 迹
        determinant = dxx * dyy - dxy * dxy # Det(H) = Dxx*Dyy - Dxy^2 行列式
        if determinant <= 1e-10: # 避免除零或负数导致的数值不稳定，如果 Det(H) 非正，说明该点的曲率估计不可靠，直接剔除。
            return False
        curvature_ratio = (trace * trace) / determinant
        threshold = ((self.edge_threshold + 1.0) ** 2) / self.edge_threshold
        return curvature_ratio < threshold # 如果主曲率比率超过阈值，说明该点更像边缘响应而不是稳定角点，需要剔除。

    def _suppress_candidates(self, candidates: list[_ScaleSpaceCandidate]) -> list[_ScaleSpaceCandidate]:
        """ 这一步属于当前实现的工程性补充，不是原始 Lowe 论文里的核心公式：
        为了避免大量响应几乎重合的点占满 max_features，这里做一个简单的空间抑制。 """
        selected: list[_ScaleSpaceCandidate] = []
        occupancy: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for candidate in candidates:
            scale_factor = float(2 ** candidate.octave)
            image_x = float(candidate.x * scale_factor)
            image_y = float(candidate.y * scale_factor)
            grid_cell = (int(image_x // 4.0), int(image_y // 4.0))
            min_distance = max(3.0, candidate.sigma * scale_factor)
            too_close = False
            for grid_y in range(grid_cell[1] - 1, grid_cell[1] + 2):
                for grid_x in range(grid_cell[0] - 1, grid_cell[0] + 2):
                    for existing_x, existing_y in occupancy.get((grid_x, grid_y), []):
                        dx = image_x - existing_x
                        dy = image_y - existing_y
                        if dx * dx + dy * dy < min_distance * min_distance:
                            too_close = True
                            break
                    if too_close:
                        break
                if too_close:
                    break
            if too_close:
                continue

            occupancy.setdefault(grid_cell, []).append((image_x, image_y))
            selected.append(candidate)
            if len(selected) >= max(self.max_features * 6, self.max_features):
                break
            
        """
        同一块强纹理附近会冒出很多非常接近的点。
        因为 _detect_candidates 先按响应从高到低排序，见 o2.py:280-281，不做 _suppress_candidates 的话，这些局部重复点会一股脑流到后面。

        会更容易把特征点预算挤满，但挤满的是“重复点”而不是“覆盖更广的点”。
        最终 keypoints 仍然会在 o2.py:143-145 被 self.max_features 截断，所以删掉抑制后，不是“得到更多有用点”，而是“更早被局部密集区域塞满”。

        运行会更慢。
        因为每个候选点后面都要做方向分配和描述子构建，流程从 o2.py:118-145 可以看到。候选点越多，这两步开销越大。

        匹配质量通常会下降，不一定是单个描述子变差，而是整体覆盖变差。
        你会得到很多来自同一角点、同一边缘附近的近重复描述子，它们彼此很像，容易造成匹配冗余和歧义；与此同时，图像其他区域本该保留下来的点反而因为数量上限进不来。

        可视化会更乱。
        同一位置附近会画出更多点和更多相似连线，信息密度上去了，但信息质量未必更高。 """
        
        return selected
        

    def _assign_orientations(self, magnitude: np.ndarray, orientation: np.ndarray, x: float, y: float, sigma: float) -> list[float]:
        """第三步：方向分配。"""

        del sigma
        # 1. 固定取关键点周围的16x16窗口。
        half_window = self.orientation_window_size // 2
        block_size = self.orientation_block_size
        window_start_y = int(round(y)) - half_window
        window_start_x = int(round(x)) - half_window
        window_end_y = window_start_y + self.orientation_window_size - 1
        window_end_x = window_start_x + self.orientation_window_size - 1
        if (
            window_start_y < 0
            or window_end_y >= magnitude.shape[0]
            or window_start_x < 0
            or window_end_x >= magnitude.shape[1]
        ):
            return []

        # 2. 把窗口切成2x2小块，为每个小块先聚合出一个梯度方向和梯度强度。
        block_votes: list[tuple[float, float, float, float]] = []
        blocks_per_side = self.orientation_window_size // block_size
        for block_y in range(blocks_per_side):
            for block_x in range(blocks_per_side):
                sample_y = window_start_y + block_y * block_size
                sample_x = window_start_x + block_x * block_size
                block_vote = self._compute_block_gradient_vote(
                    magnitude,
                    orientation,
                    sample_x,
                    sample_y,
                    block_size,
                )
                if block_vote is None:
                    continue
                block_magnitude, block_angle = block_vote
                block_center_x = sample_x + (block_size - 1) / 2.0
                block_center_y = sample_y + (block_size - 1) / 2.0
                block_votes.append((block_magnitude, block_angle, block_center_x, block_center_y))

        if not block_votes:
            return []

        histogram = np.zeros(36, dtype=np.float32)
        local_max_magnitude = max(vote[0] for vote in block_votes)
        # 3. 丢弃过弱的小块梯度，避免弱纹理和噪声扰动主方向。
        weak_gradient_floor = max(1e-6, self.orientation_magnitude_ratio * local_max_magnitude)
        window_sigma = max(1.0, self.orientation_window_size / 2.0)
        # 4. 将剩余小块的高斯加权梯度幅值累积到36-bin方向直方图中。
        for sample_magnitude, sample_angle, sample_x, sample_y in block_votes:
            if sample_magnitude < weak_gradient_floor:
                continue
            dx = float(sample_x - x)
            dy = float(sample_y - y)
            # 方向分配阶段先聚合2x2块梯度，再做弱梯度剔除和高斯加权累积
            weight = math.exp(-(dx * dx + dy * dy) / (2.0 * window_sigma * window_sigma))
            bin_index = int(sample_angle // 10.0) % 36
            histogram[bin_index] += weight * sample_magnitude

        # 对环形直方图做平滑，降低量化噪声对主方向选择的影响。
        histogram = self._smooth_circular_histogram(histogram, passes=4)
        return self._extract_orientation_peaks(histogram, peak_ratio=0.8)

    def _compute_block_gradient_vote(self, magnitude: np.ndarray, orientation: np.ndarray, start_x: int, start_y: int, block_size: int) -> tuple[float, float] | None:
        block_magnitude = magnitude[start_y : start_y + block_size, start_x : start_x + block_size]
        block_orientation = orientation[start_y : start_y + block_size, start_x : start_x + block_size]
        if block_magnitude.size == 0:
            return None
        # 先把 block 内所有像素的梯度向量按角度分解成 x 和 y 分量，再求和得到 block 的整体梯度向量，最后从整体梯度向量计算出一个代表性的幅值和方向
        block_orientation_rad = np.deg2rad(block_orientation)
        grad_x = float(np.sum(block_magnitude * np.cos(block_orientation_rad)))
        grad_y = float(np.sum(block_magnitude * np.sin(block_orientation_rad)))
        combined_magnitude = math.hypot(grad_x, grad_y)
        if combined_magnitude <= 1e-8:
            return None
        combined_angle = math.degrees(math.atan2(grad_y, grad_x)) % 360.0
        return combined_magnitude, combined_angle

    def _extract_orientation_peaks(self, histogram: np.ndarray, peak_ratio: float) -> list[float]:
        max_value = float(np.max(histogram, initial=0.0))
        if max_value <= 1e-8:
            return []

        threshold = max_value * peak_ratio
        peak_angles: list[float] = []
        bin_count = histogram.shape[0]
        for bin_index in range(bin_count):
            left = float(histogram[(bin_index - 1) % bin_count])
            center = float(histogram[bin_index])
            right = float(histogram[(bin_index + 1) % bin_count])
            if center < threshold:
                continue
            if center < left or center < right:
                continue

            denominator = left - 2.0 * center + right
            peak_offset = 0.0 if abs(denominator) < 1e-8 else 0.5 * (left - right) / denominator
            peak_offset = float(np.clip(peak_offset, -0.5, 0.5))
            angle = ((bin_index + peak_offset) * 360.0 / bin_count) % 360.0
            peak_angles.append(float(angle))

        peak_angles.sort()
        return peak_angles

    def _build_descriptor(self, magnitude: np.ndarray, orientation: np.ndarray, x: float, y: float, sigma: float, angle: float ) -> np.ndarray | None:
        """第四步：关键点描述子生成。"""
        # 1. 先按关键点的主方向和尺度把邻域归一化到16x16窗口。
        sigma_safe = max(1.0, sigma)
        half_window = self.descriptor_window_size / 2.0
        block_size = self.descriptor_window_size / float(self.descriptor_block_count)
        orientation_bin_width = 360.0 / float(self.descriptor_orientation_bins)
        gaussian_variance = self.descriptor_window_size / 2.0
        radius = max(8, int(round(half_window * sigma_safe)))
        min_y = int(math.floor(y - radius))
        max_y = int(math.ceil(y + radius))
        min_x = int(math.floor(x - radius))
        max_x = int(math.ceil(x + radius))
        if (
            min_y < 1
            or max_y >= magnitude.shape[0] - 1
            or min_x < 1
            or max_x >= magnitude.shape[1] - 1
        ):
            return None

        descriptor = np.zeros(
            (self.descriptor_block_count, self.descriptor_block_count, self.descriptor_orientation_bins),
            dtype=np.float32,
        )
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        for yy in range(min_y, max_y + 1):
            for xx in range(min_x, max_x + 1):
                dx = float(xx - x)
                dy = float(yy - y)
                rotated_x = (cos_angle * dx + sin_angle * dy) / sigma_safe
                rotated_y = (-sin_angle * dx + cos_angle * dy) / sigma_safe
                if abs(rotated_x) >= half_window or abs(rotated_y) >= half_window:
                    continue
                # 2. 再把这个16x16窗口切成4x4个子区域，每个子区域统计8个方向bin。
                block_x = int((rotated_x + half_window) // block_size)
                block_y = int((rotated_y + half_window) // block_size)
                if (
                    block_x < 0
                    or block_x >= self.descriptor_block_count
                    or block_y < 0
                    or block_y >= self.descriptor_block_count
                ):
                    continue
                relative_angle = (float(orientation[yy, xx]) - angle) % 360.0
                orientation_bin = int(relative_angle // orientation_bin_width) % self.descriptor_orientation_bins
                # 3. 梯度幅值先乘一个高斯权重，方差取半个窗口宽度，让中心区域平滑地主导描述子。
                gaussian_weight = math.exp(
                    -(rotated_x * rotated_x + rotated_y * rotated_y) / (2.0 * gaussian_variance)
                )
                sample_weight = gaussian_weight * float(magnitude[yy, xx])
                if sample_weight <= 1e-8:
                    continue
                descriptor[block_y, block_x, orientation_bin] += sample_weight

        vector = descriptor.reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return None
        # 4. 最后把4x4x8拼成128维向量，做L2归一化、0.2截断，再次归一化。
        vector = vector / norm
        vector = np.clip(vector, 0.0, self.descriptor_clip_value)
        renorm = float(np.linalg.norm(vector))
        if renorm < 1e-8:
            return None
        return (vector / renorm).astype(np.float32)

    def _smooth_circular_histogram(self, histogram: np.ndarray, passes: int) -> np.ndarray:
        smoothed = histogram.astype(np.float32, copy=True)
        for _ in range(passes):
            smoothed = (
                np.roll(smoothed, 1)
                + 2.0 * smoothed
                + np.roll(smoothed, -1)
            ) / 4.0
        return smoothed


def read_metrics(metrics_file: Path) -> list[MetricRow]:
    """读取 O2 历史指标。"""
    # 第一次运行时指标文件还不存在，这里直接返回空列表，让上层按“无历史结果”处理即可。
    if not metrics_file.exists():
        return []

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[MetricRow] = []
        for row in reader:
            # scene 是后面做按场景合并的主键；没有 scene 的行基本可以视为脏数据。
            if not row.get("scene"):
                continue
            # 只保留当前 O2 关心的字段，避免旧文件里混入无关列时影响后续写回格式。
            rows.append({key: row.get(key, "") for key in METRIC_FIELDNAMES})
        return rows


def write_metrics(metrics_file: Path, rows: list[MetricRow]) -> None:
    """按场景合并并回写 O2 指标。"""
    # O2 是按场景重跑的：同一个 scene 再跑一次时，希望新结果覆盖旧结果，
    # 没有重跑的 scene 则保持原样，所以这里不是简单追加，而是做一次按 scene 的 merge。
    existing_rows = read_metrics(metrics_file)
    rows_by_scene = {str(row["scene"]): row for row in rows}
    merged_rows: list[MetricRow] = []

    for row in existing_rows:
        scene_name = str(row["scene"])
        replacement = rows_by_scene.pop(scene_name, None)
        merged_rows.append(replacement if replacement is not None else row)

    # 走到这里，rows_by_scene 里剩下的是“旧文件里没有、这次新产生”的场景。
    merged_rows.extend(rows_by_scene.values())

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(merged_rows)


def create_sift_detector(max_features: int, contrast_threshold: float) -> ManualSiftDetector:
    """创建手写 SIFT 检测器。"""

    return ManualSiftDetector(max_features=max_features, contrast_threshold=contrast_threshold)


def draw_keypoints(image_bgr: Any, keypoints: list[ManualKeypoint]) -> Any:
    """把关键点以可视化形式画到图像上，便于结果检查。"""

    canvas = np.asarray(image_bgr).copy()
    # 只画响应最强的少量关键点，避免可视化结果被过密标记淹没。
    visible_keypoints = sorted(keypoints, key=lambda keypoint: keypoint.response, reverse=True)[:MAX_VISUALIZED_SIFT_ITEMS]
    for keypoint in visible_keypoints:
        center = (int(round(keypoint.x)), int(round(keypoint.y)))
        radius = max(2, int(round(keypoint.size / 2.0)))
        angle_rad = math.radians(keypoint.angle)
        tip = (
            int(round(center[0] + radius * math.cos(angle_rad))),
            int(round(center[1] + radius * math.sin(angle_rad))),
        )
        cv2.circle(canvas, center, radius, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(canvas, center, tip, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    return canvas


def draw_matches(
    left_bgr: Any,
    left_keypoints: list[ManualKeypoint],
    right_bgr: Any,
    right_keypoints: list[ManualKeypoint],
    matches: list[ManualMatch],
) -> Any:
    """把匹配结果画到拼接画布上，便于人工检查。"""

    left = np.asarray(left_bgr)
    right = np.asarray(right_bgr)
    output_height = max(left.shape[0], right.shape[0])
    output_width = left.shape[1] + right.shape[1]
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right

    x_offset = left.shape[1]
    # 只展示距离最小的少量匹配，便于人工观察每条连线。
    visible_matches = sorted(matches, key=lambda match: match.distance)[:MAX_VISUALIZED_SIFT_ITEMS]
    for index, match in enumerate(visible_matches):
        left_point = left_keypoints[match.queryIdx]
        right_point = right_keypoints[match.trainIdx]
        color = (
            int((37 * index) % 192 + 48),
            int((83 * index) % 192 + 48),
            int((131 * index) % 192 + 48),
        )
        start = (int(round(left_point.x)), int(round(left_point.y)))
        end = (int(round(right_point.x + x_offset)), int(round(right_point.y)))
        cv2.circle(canvas, start, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, end, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, start, end, color, 1, lineType=cv2.LINE_AA)
    return canvas


def _knn_match_descriptors(left_descriptors: np.ndarray, right_descriptors: np.ndarray, k: int) -> list[list[ManualMatch]]:
    if left_descriptors.size == 0 or right_descriptors.size == 0:
        return []

    neighbor_count = max(1, min(int(k), right_descriptors.shape[0]))
    left = np.asarray(left_descriptors, dtype=np.float32)
    right = np.asarray(right_descriptors, dtype=np.float32)
    distance_squared = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    distance_squared = np.maximum(distance_squared, 0.0, dtype=np.float32)
    distances = np.sqrt(distance_squared, dtype=np.float32)
    nearest_indices = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]

    knn_matches: list[list[ManualMatch]] = []
    for query_index in range(distances.shape[0]):
        candidate_indices = nearest_indices[query_index]
        ordered_indices = candidate_indices[np.argsort(distances[query_index, candidate_indices])]
        knn_matches.append(
            [
                ManualMatch(
                    queryIdx=query_index,
                    trainIdx=int(train_index),
                    distance=float(distances[query_index, train_index]),
                )
                for train_index in ordered_indices.tolist()
            ]
        )
    return knn_matches


def _ratio_filtered_matches(knn_matches: list[list[ManualMatch]], ratio_test: float) -> list[ManualMatch]:
    """对 KNN 匹配应用 Lowe ratio test。"""
    filtered: list[ManualMatch] = []
    for pair in knn_matches:
        # 正常情况下 knnMatch(k=2) 会返回两个候选；极少数边界情况不足两个时，
        # 没法做 ratio test，直接跳过最稳妥。
        if len(pair) < 2:
            continue
        first, second = pair
        # 最近邻必须显著优于次近邻，才认为这个描述符匹配“足够明确”。
        if first.distance < ratio_test * second.distance:
            filtered.append(first)
    return filtered


def _mutual_ratio_matches(
    left_descriptors: np.ndarray,
    right_descriptors: np.ndarray,
    ratio_test: float,
) -> tuple[int, list[ManualMatch], list[ManualMatch]]:
    """先做双向 KNN+ratio test，再取互为匹配的描述子对，减少偶然误匹配。"""

    forward_knn = _knn_match_descriptors(left_descriptors, right_descriptors, k=2)
    backward_knn = _knn_match_descriptors(right_descriptors, left_descriptors, k=2)
    forward_ratio = _ratio_filtered_matches(forward_knn, ratio_test)
    backward_ratio = _ratio_filtered_matches(backward_knn, ratio_test)
    # 反向保留下来的匹配先编码成索引对，后面判断“是否互为最近邻”会更直接。
    backward_pairs = {(match.trainIdx, match.queryIdx) for match in backward_ratio}
    mutual = [match for match in forward_ratio if (match.queryIdx, match.trainIdx) in backward_pairs]
    # 为了让可视化和调试输出更稳定，这里按距离从小到大排序。
    mutual.sort(key=lambda match: match.distance)
    return len(forward_knn), forward_ratio, mutual


def _geometric_inlier_matches(
    left_keypoints: list[ManualKeypoint],
    right_keypoints: list[ManualKeypoint],
    matches: list[ManualMatch],
) -> tuple[list[ManualMatch], int]:
    """使用基础矩阵 RANSAC 剔除几何上明显不合理的匹配。"""

    # 基础矩阵最少需要 8 对点；样本太少时强行做 RANSAC 只会让结果更不稳定。
    if len(matches) < 8:
        return matches, 0

    # OpenCV 这里要求输入形状是 N x 1 x 2，所以要先把 KeyPoint.pt 摘出来再 reshape。
    left_points = np.asarray([left_keypoints[match.queryIdx].pt for match in matches], dtype=np.float32).reshape(-1, 1, 2)
    right_points = np.asarray([right_keypoints[match.trainIdx].pt for match in matches], dtype=np.float32).reshape(-1, 1, 2)
    fundamental, inlier_mask = cv2.findFundamentalMat(left_points, right_points, cv2.FM_RANSAC, 2.5, 0.99)
    # 估计失败时，宁可保留原匹配，也不要伪造一个“清洗后”的结果。
    if fundamental is None or inlier_mask is None:
        return matches, 0

    mask = np.asarray(inlier_mask, dtype=np.uint8).reshape(-1) > 0
    inliers = [match for match, keep in zip(matches, mask) if keep]
    # 即便 RANSAC 返回了 mask，如果最终内点少于 8 个，也说明几何约束不够可信。
    if len(inliers) < 8:
        return matches, 0
    return inliers, int(mask.sum())


def _select_geometry_aware_matches(
    left_keypoints: list[ManualKeypoint],
    right_keypoints: list[ManualKeypoint],
    ratio_filtered: list[ManualMatch],
    mutual_matches: list[ManualMatch],
) -> tuple[list[ManualMatch], int]:
    """结合 mutual matching 和 RANSAC 结果选择更稳健的匹配集合。"""
    # 默认从单向 ratio test 结果开始，因为这是最宽松、召回率最高的候选集。
    candidate_matches = ratio_filtered
    mutual_retention = (len(mutual_matches) / len(ratio_filtered)) if ratio_filtered else 0.0
    # 当 mutual matching 保留了足够多的点，而且保留比例也够高时，
    # 说明双向一致性没有把样本削得太狠，这时优先用 mutual 结果更稳。
    if len(mutual_matches) >= 24 and mutual_retention >= 0.75:
        candidate_matches = mutual_matches

    # 候选集太小时不做 RANSAC，因为几何模型会被少量偶然点严重影响。
    if len(candidate_matches) < 24:
        return candidate_matches, 0

    geometric_matches, ransac_inliers = _geometric_inlier_matches(left_keypoints, right_keypoints, candidate_matches)
    # 只有当 RANSAC 之后还剩下“数量够多、比例也不离谱”的内点时，才相信几何筛选真的起了正作用。
    if len(geometric_matches) >= max(24, int(len(candidate_matches) * 0.5)):
        return geometric_matches, ransac_inliers
    return candidate_matches, ransac_inliers


def _project_point(homography: Any, x: float, y: float) -> tuple[float, float] | None:
    """把图像点通过单应矩阵投影到变换后坐标系；若齐次坐标退化则返回 None。"""

    # 这里显式构造成齐次坐标，便于直接和 3x3 homography 做矩阵乘法。
    point = np.asarray([x, y, 1.0], dtype=np.float64)
    projected = homography @ point
    # 分母接近 0 说明投影退化了；继续相除只会放大数值误差。
    if abs(float(projected[2])) < 1e-8:
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def _rng_for_scene(scene_name: str) -> tuple[int, Any]:
    """为每个场景稳定地产生随机种子，保证多次运行结果可复现。"""

    # 不直接用 Python 内建 hash，是因为它默认带随机扰动；这里改用 sha256 保证跨进程稳定。
    seed = int(hashlib.sha256(scene_name.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    return seed, np.random.default_rng(seed)


def _transform_family(scene_name: str, scene_index: int) -> str:
    """稳定地为场景挑选一种变换族。"""
    # family 和 transform 本身分开播种，是为了避免“改了某一类变换的参数采样逻辑后，
    # 连场景落到哪一类 family 都跟着漂移”。
    _, rng = _rng_for_scene(f"family::{scene_index}::{scene_name}")
    return str(rng.choice(DISTINCT_TRANSFORM_FAMILIES))


def _rotation_matrix(width: int, height: int, angle_deg: float, scale: float) -> Any:
    """构造绕图像中心旋转/缩放的 3x3 单应矩阵。"""

    # OpenCV 返回的是 2x3 仿射矩阵，这里手动扩成 3x3，后面做点投影时会统一很多。
    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, angle_deg, scale)
    homography = np.eye(3, dtype=np.float32)
    homography[:2, :] = affine
    return homography


def _affine_homography(width: int, height: int, rng: Any) -> tuple[Any, dict[str, float]]:
    """随机生成仿射风格的单应矩阵及其参数说明。"""

    # 这里把角度、尺度、剪切、平移拆开采样，目的是让 README / metrics 里能把变换说清楚，
    # 后面如果某一类变化导致 repeatability 明显下降，也容易定位是哪个因素更敏感。
    angle_deg = float(rng.uniform(-16.0, 16.0))
    scale = float(rng.uniform(0.88, 1.12))
    shear_x = float(rng.uniform(-0.08, 0.08))
    shear_y = float(rng.uniform(-0.05, 0.05))
    tx = float(rng.uniform(-0.06, 0.06) * width)
    ty = float(rng.uniform(-0.06, 0.06) * height)

    angle = np.deg2rad(angle_deg)
    # 下面这几块矩阵是按“先把中心挪到原点，再做 rotation/scale/shear，
    # 最后挪回图像中心并叠加平移”的思路拼起来的。
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scaling = np.array(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    shear = np.array(
        [
            [1.0, shear_x, 0.0],
            [shear_y, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    center_to_origin = np.array(
        [
            [1.0, 0.0, -width / 2.0],
            [0.0, 1.0, -height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    origin_to_center = np.array(
        [
            [1.0, 0.0, width / 2.0 + tx],
            [0.0, 1.0, height / 2.0 + ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    homography = origin_to_center @ shear @ scaling @ rotation @ center_to_origin
    return homography, {
        "angle_deg": angle_deg,
        "scale": scale,
        "shear_x": shear_x,
        "shear_y": shear_y,
        "tx_px": tx,
        "ty_px": ty,
    }


def _apply_intensity_variation(image_bgr: Any, rng: Any) -> tuple[Any, dict[str, float]]:
    """只改变亮度/对比度/噪声/模糊，不改变几何结构，用来测试 SIFT 对光照变化的鲁棒性。"""

    # alpha/beta/gamma 分别覆盖线性对比度、整体亮度和非线性亮度响应；
    # 再叠一点噪声和模糊，能更接近真实采集条件里的曝光/成像扰动。
    alpha = float(rng.uniform(0.7, 1.35))
    beta = float(rng.uniform(-32.0, 32.0))
    gamma = float(rng.uniform(0.75, 1.35))
    noise_sigma = float(rng.uniform(0.0, 8.0))
    blur_kernel = int(rng.choice([0, 3, 5]))

    # 先转到 [0, 1] 浮点域处理，再回写成 uint8，这样组合多个强度操作时更可控。
    working = image_bgr.astype(np.float32) / 255.0
    working = np.clip(working * alpha + beta / 255.0, 0.0, 1.0)
    working = np.power(working, gamma)
    if noise_sigma > 0.0:
        working += rng.normal(0.0, noise_sigma / 255.0, size=working.shape)
    working = np.clip(working, 0.0, 1.0)
    transformed = np.rint(working * 255.0).astype(np.uint8)
    if blur_kernel > 0:
        transformed = cv2.GaussianBlur(transformed, (blur_kernel, blur_kernel), 0)
    return transformed, {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "noise_sigma": noise_sigma,
        "blur_kernel": blur_kernel,
    }


def generate_transformed_view(image_bgr: Any, scene_name: str, scene_index: int) -> tuple[Any, Any, str, int, TransformParams]:
    """根据场景稳定随机地产生一种变换后的右图，并返回对应的真实几何/光照参数。"""

    height, width = image_bgr.shape[:2]
    seed, rng = _rng_for_scene(f"transform::{scene_index}::{scene_name}")
    family = _transform_family(scene_name, scene_index)

    if family == "rotation":
        # rotation 家族只做“绕中心旋转/缩放 + 小范围平移”，属于比较直观的相机姿态扰动。
        angle_deg = float(rng.uniform(-28.0, 28.0))
        scale = float(rng.uniform(0.82, 1.18))
        tx = float(rng.uniform(-0.03, 0.03) * width)
        ty = float(rng.uniform(-0.03, 0.03) * height)
        homography = _rotation_matrix(width, height, angle_deg, scale)
        homography[0, 2] += tx
        homography[1, 2] += ty
        transformed = cv2.warpPerspective(
            image_bgr,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        params: TransformParams = {
            "family": family,
            "angle_deg": angle_deg,
            "scale": scale,
            "tx_px": tx,
            "ty_px": ty,
        }
        return transformed, homography, family, seed, params

    if family == "affine":
        # affine 家族比单纯旋转更激进一些，额外引入 shear，能测试关键点在形变下的稳定性。
        homography, affine_params = _affine_homography(width, height, rng)
        transformed = cv2.warpPerspective(
            image_bgr,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        params: TransformParams = {
            "family": family,
            "angle_deg": affine_params["angle_deg"],
            "scale": affine_params["scale"],
            "shear_x": affine_params["shear_x"],
            "shear_y": affine_params["shear_y"],
            "tx_px": affine_params["tx_px"],
            "ty_px": affine_params["ty_px"],
        }
        return transformed, homography, family, seed, params

    # intensity 家族不改几何，所以真实 homography 就是单位阵；
    # 这样后面 repeatability 下降时，基本可以直接归因到描述符对光照扰动的抗性不够。
    transformed, intensity_params = _apply_intensity_variation(image_bgr, rng)
    homography = np.eye(3, dtype=np.float32)
    params: TransformParams = {
        "family": family,
        "alpha": intensity_params["alpha"],
        "beta": intensity_params["beta"],
        "gamma": intensity_params["gamma"],
        "noise_sigma": intensity_params["noise_sigma"],
        "blur_kernel": intensity_params["blur_kernel"],
    }
    return transformed, homography, family, seed, params


def _repeatable_matches(original_keypoints: Any, transformed_keypoints: Any, ratio_matches: list[Any], homography: Any, threshold_px: float) -> list[Any]:
    repeatable: list[Any] = []
    for match in ratio_matches:
        # queryIdx 指向原图关键点，trainIdx 指向变换后图像里的候选对应点。
        source = original_keypoints[match.queryIdx].pt
        target = transformed_keypoints[match.trainIdx].pt
        projected = _project_point(homography, float(source[0]), float(source[1]))
        if projected is None:
            continue
        dx = projected[0] - float(target[0])
        dy = projected[1] - float(target[1])
        # 这里的判据很直接：描述符匹配到的 target 点，如果离真实投影位置足够近，
        # 就把它记成“可重复匹配”。
        if (dx * dx + dy * dy) ** 0.5 <= threshold_px:
            repeatable.append(match)
    # 同样按距离排序，保证图像可视化时优先画出更可靠的匹配。
    repeatable.sort(key=lambda item: item.distance)
    return repeatable


def validate_results(keypoints_dir: Path, matches_dir: Path, metrics_file: Path, scene_name: str | None = None) -> int:
    """验证 O2 输出文件是否存在。"""
    print(f"Validating O2 keypoint directory: {keypoints_dir}")
    print(f"Validating O2 match directory: {matches_dir}")
    print(f"Validating O2 metrics file: {metrics_file}")

    if not metrics_file.exists():
        print(f"O2 metrics file not found: {metrics_file}", file=sys.stderr)
        return 1

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("scene")]

    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
        rows = [row for row in rows if row.get("scene") == scene_name]

    # 校验逻辑以 metrics.csv 为入口，因为 metrics 代表“哪些 scene 被认为已经成功跑过”。
    if not rows:
        print("No O2 metric rows found for validation.", file=sys.stderr)
        return 1

    missing_any = False
    for row in rows:
        current_scene = row["scene"]
        keypoint_scene_dir = keypoints_dir / current_scene
        match_scene_dir = matches_dir / current_scene
        # O2 最核心的产物就是两张关键点图和一张匹配图；README 丢了虽然不理想，
        # 但通常不影响后续查结果，所以这里先把“最低限度必须存在的文件”收紧到这三项。
        required_paths = (
            keypoint_scene_dir / "im0_keypoints.png",
            keypoint_scene_dir / "im1_keypoints.png",
            match_scene_dir / "sift_matches.png",
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            missing_any = True
            print(f"[MISSING] {current_scene}: " + ", ".join(missing))
        else:
            print(f"[OK] {current_scene}: required O2 files present")

    if missing_any:
        print("Validation status: issues found")
        return 1

    print("Validation status: all checked O2 scene folders contain the expected files")
    return 0


def run(
    repo_root: Path,
    middlebury_root: Path,
    config: O2Config,
    max_scenes: int | None,
    dry_run: bool,
    scene_name: str | None,
) -> int:
    # “总共发现多少 scene”“这次实际会处理多少 scene”
    scenes = discover_scenes(middlebury_root)
    scenes_count = len(scenes)
    scenes = filter_scene_dirs(scenes, scene_name)

    if max_scenes is not None:
        # 负数没有实际意义，直接按参数错误返回，比悄悄当作 0 或 None 更容易排查。
        if max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[:max_scenes]  #直接截断列表，保证处理顺序稳定。
        
    print(f"Repository root: {repo_root}") # 项目的根目录路径
    print(f"Middlebury root: {middlebury_root}") # Middlebury 数据集的根目录
    print(f"Discovered scenes with im0.png/im1.png: {scenes_count}") # 实际在数据集目录下识别到的有效场景文件夹
    print(f"O2 keypoint output dir: {config.keypoints_dir}") # 关键点检测结果的输出目录
    print(f"O2 match output dir: {config.matches_dir}") # 匹配结果的输出目录
    print(f"O2 metrics file: {config.metrics_file}") # 所有场景评估指标的 CSV 文件路径
    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
    print("Scenes to process: " + ", ".join(scene.name for scene in scenes) if scenes else "Scenes to process: none")
    
    #================================数据集目录检查================================

    #判定数据集指定目录是否存在
    if not middlebury_root.exists():
        # dry-run / discovery-only 模式下，跳过检查
        if dry_run or max_scenes == 0:
            print("Middlebury root does not exist yet. Discovery-only mode completed without processing.")
            return 0
        print(
            f"Middlebury root not found: {middlebury_root}\n"
            "Place dataset scenes there, or rerun with --dry-run or --max-scenes 0.",
            file=sys.stderr,
        )
        return 1

    # 目录存在但没有任何合法 scene 时
    if scenes_count == 0:
        # dry-run / discovery-only 模式下，跳过检查
        if dry_run or max_scenes == 0:
            print("No valid scenes found. Discovery-only mode completed without processing.")
            return 0
        print(
            f"No scene directories containing im0.png and im1.png were found under {middlebury_root}.\n"
            "Add Middlebury scenes, or rerun with --dry-run or --max-scenes 0.",
            file=sys.stderr,
        )
        return 1

    # 指定的 scene_name 不存在
    if scene_name is not None and not scenes:
        print(f"No discovered scenes matched --scene-name {scene_name!r} under {middlebury_root}.", file=sys.stderr)
        return 1

    # 到这里说明发现逻辑都正常；如果只是 dry-run，就不再创建目录也不做任何图像计算。
    if dry_run or max_scenes == 0:
        print("Dry run requested; no outputs were written.")
        return 0
    
    #===============================核心处理流程================================

    # detector 在整个 O2 过程中复用即可，没有必要为每个 scene 单独创建。
    detector = create_sift_detector(config.max_features, config.contrast_threshold)
    # 输出目录确认
    config.keypoints_dir.mkdir(parents=True, exist_ok=True)
    config.matches_dir.mkdir(parents=True, exist_ok=True)

    # metric_rows 保存每个 scene 的评估结果，最后会写到 CSV 里
    metric_rows: list[dict[str, str | int | float]] = []
    # 4 像素阈值不是 SIFT 算法本身的参数，而是这里定义“几何上算 repeatable”的工程判据。
    threshold_px = 4.0

    for scene_index, scene_dir in enumerate(scenes):
        # O2 的左图固定取原始 im0
        # 而是由左图生成一个带可控扰动的 transformed view
        original_path = scene_dir / "im0.png"
        original_bgr = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        if original_bgr is None:
            raise FileNotFoundError(f"Could not read image: {original_path}")
        original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY) # 原图转灰度
        transformed_bgr, homography, transform_family, random_seed, transform_params = generate_transformed_view(
            original_bgr,
            scene_dir.name,
            scene_index,
        )
        # 转换图像同样转灰度，准备后续 SIFT 处理
        transformed_gray = cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2GRAY)

        # 在整张 original_gray 上执行一次完整 SIFT；第二个参数传 None，表示不额外提供 mask，
        # 所以 OpenCV 会直接在整幅图像内完成检测和描述。
        # 返回的 original_keypoints 是“已经走完 SIFT 前 3 步”的结果：
        # 1) 先在 original_gray 的尺度空间 / DoG 金字塔里找极值候选点；
        # 2) 再结合 create_sift_detector(...) 里设定的 contrastThreshold 做低对比度剔除，
        #    同时过滤边缘响应并完成亚像素级精定位；
        # 3) 然后给每个保留下来的关键点分配主方向，因此 keypoint 的 pt / size / angle
        #    在这里都已经是可直接使用的稳定属性。
        # 返回的 original_descriptors 则对应 SIFT 第 4 步：围绕每个 original_keypoints 统计局部梯度，
        # 为每个关键点生成一个 128 维描述符，供后面的手写 L2 KNN 匹配使用。
        original_keypoints, original_descriptors = detector.detectAndCompute(original_gray, None)
        # 对变换后的 transformed_gray 重复同一套 detect + compute 流程，得到另一组
        # transformed_keypoints / transformed_descriptors；下面的 repeatability 评估
        # 就是专门比较这两次 detectAndCompute(...) 的输出是否还能对应上。
        transformed_keypoints, transformed_descriptors = detector.detectAndCompute(transformed_gray, None)

        # 先把匹配统计初始化成“没有匹配”的状态，这样即便某个 scene 一个描述符都没提到，
        # 后面生成 README / CSV 时也还是能写出完整记录。
        raw_matches = 0
        ratio_matches = []
        repeatable_matches = []
        if (
            original_descriptors is not None
            and transformed_descriptors is not None
            and len(original_descriptors) > 0
            and len(transformed_descriptors) > 0
        ):
            # 这里开始不再属于 SIFT 本体，而是对第 4 步产出的描述符做项目级评估。
            # 先对 128 维描述符做 KNN 最近邻匹配，建立候选对应关系。
            knn_matches = _knn_match_descriptors(original_descriptors, transformed_descriptors, k=2)
            raw_matches = len(knn_matches)
            # Lowe ratio test 用来剔除“最近邻和次近邻太接近”的歧义匹配。
            ratio_matches = _ratio_filtered_matches(knn_matches, float(config.ratio_test))
            # 再结合已知随机变换的真实几何关系，筛出真正可重复的关键点对应。
            repeatable_matches = _repeatable_matches(
                original_keypoints,
                transformed_keypoints,
                ratio_matches,
                homography,
                threshold_px,
            )

        original_count = len(original_keypoints)
        transformed_count = len(transformed_keypoints)
        # repeatability 采用 min(left, right) 归一化，是为了避免某一边关键点特别多时
        # 把分母拉得过大；这里更关心“较小那一侧里有多少点还能稳定找回来”。
        repeatability = (
            float(len(repeatable_matches) / min(original_count, transformed_count))
            if min(original_count, transformed_count) > 0
            else 0.0
        )

        # O2 把关键点可视化和匹配可视化分到两个目录，是为了让“检测质量”和“匹配质量”
        # 可以分开检查，不至于所有产物都堆在一个场景文件夹里。
        keypoint_scene_dir = config.keypoints_dir / scene_dir.name
        match_scene_dir = config.matches_dir / scene_dir.name
        keypoint_scene_dir.mkdir(parents=True, exist_ok=True)
        match_scene_dir.mkdir(parents=True, exist_ok=True)

        # 这两张图主要用于人工检查：关键点是否覆盖了有纹理的区域、变换后是否仍有足够响应。
        cv2.imwrite(str(keypoint_scene_dir / "im0_keypoints.png"), draw_keypoints(original_bgr, original_keypoints))
        cv2.imwrite(str(keypoint_scene_dir / "im1_keypoints.png"), draw_keypoints(transformed_bgr, transformed_keypoints))
        # README 则把“这一 scene 是怎么变出来的、参数是多少、关键点数量是多少”落成文本，
        # 后面回看单个场景时不必再去翻 metrics.csv。
        write_scene_text(
            keypoint_scene_dir / "README.txt",
            [
                f"scene: {scene_dir.name}",
                "generator: O2 SIFT repeatability baseline on original image plus random transformation",
                "evaluation_definition: detect SIFT on original image before transformation, detect again after a randomly generated transformation, then keep only descriptor matches consistent with the known random transform",
                f"transform_family: {transform_family}",
                f"random_seed: {random_seed}",
                f"transform_params_json: {json.dumps(transform_params, sort_keys=True)}",
                f"im0_keypoints: {original_count}",
                f"im1_keypoints: {transformed_count}",
                f"max_features: {config.max_features}",
                f"contrast_threshold: {config.contrast_threshold}",
            ],
        )

        # 匹配图太密会完全看不清，所以这里只画最可靠的前若干条 repeatable matches。
        draw_count = min(len(repeatable_matches), config.max_draw_matches)
        match_image = draw_matches(
            original_bgr,
            original_keypoints,
            transformed_bgr,
            transformed_keypoints,
            repeatable_matches[:draw_count],
        )
        cv2.imwrite(str(match_scene_dir / "sift_matches.png"), match_image)
        # 匹配 README 更偏向评估视角：这次总共匹配了多少、ratio 之后剩多少、几何上真正 repeatable 的有多少。
        write_scene_text(
            match_scene_dir / "README.txt",
            [
                f"scene: {scene_dir.name}",
                "generator: O2 SIFT repeatability baseline on original image plus random transformation",
                f"transform_family: {transform_family}",
                f"random_seed: {random_seed}",
                f"transform_params_json: {json.dumps(transform_params, sort_keys=True)}",
                f"raw_knn_matches: {raw_matches}",
                f"ratio_filtered_matches: {len(ratio_matches)}",
                f"repeatable_matches: {len(repeatable_matches)}",
                f"homography_threshold_px: {threshold_px}",
                f"ratio_test: {config.ratio_test}",
                f"drawn_matches: {draw_count}",
                "repeatability: repeatable_matches / min(original_keypoints, transformed_keypoints)",
            ],
        )

        # CSV 里的字段尽量保持原子化，方便后面直接 pandas 读进来做统计或画图。
        metric_rows.append(
            {
                "scene": scene_dir.name,
                "transform_family": transform_family,
                "random_seed": random_seed,
                "left_keypoints": original_count,
                "right_keypoints": transformed_count,
                "raw_knn_matches": raw_matches,
                "ratio_test_matches": len(ratio_matches),
                "repeatable_matches": len(repeatable_matches),
                "repeatability": f"{repeatability:.6f}",
                "repeatability_proxy": f"{repeatability:.6f}",
                "homography_threshold_px": f"{threshold_px:.2f}",
                "transform_params_json": json.dumps(transform_params, sort_keys=True),
            }
        )
        # 每个 scene 处理完就打印一行摘要，跑长任务时不用等到最后也能知道有没有明显异常。
        print(
            f"Wrote O2 scene: {scene_dir.name} "
            f"(family={transform_family}, seed={random_seed}, original={original_count}, transformed={transformed_count}, "
            f"ratio={len(ratio_matches)}, repeatable={len(repeatable_matches)}, repeatability={repeatability:.6f})"
        )

    # 整批 scene 全部处理完后，再统一写 metrics summary，保证文件内容总是对应一轮完整运行。
    write_metrics(config.metrics_file, metric_rows)
    print(f"Wrote O2 metrics summary: {config.metrics_file}")
    return 0


if __name__ == '__main__':
    # 统一走 run_objective_entry
    from entry_utils import run_objective_entry

    raise SystemExit(run_objective_entry('o2', __file__))
