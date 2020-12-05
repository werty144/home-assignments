#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple
from collections import namedtuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    eye3x4,
    build_correspondences,
    _remove_correspondences_with_ids,
    triangulate_correspondences,
    TriangulationParameters
)

from camera_track import CameraTracker


def find_best_start_poses(frames_n, corner_storage, intrinsic_mat):
    best_frames = (0, 0)
    best_snd_pose = None
    best_pose_score = 0
    for first_frame in range(frames_n):
        for second_frame in range(first_frame + 5, frames_n):
            pose, pose_score = get_pose_with_score(first_frame, second_frame, corner_storage, intrinsic_mat)
            if pose_score > best_pose_score:
                best_frames = (first_frame, second_frame)
                best_snd_pose = pose
                best_pose_score = pose_score
    return best_frames, view_mat3x4_to_pose(eye3x4()), best_snd_pose


def get_pose_with_score(frame_1, frame_2, corner_storage, intrinsic_mat):
    corners1, corners2 = corner_storage[frame_1], corner_storage[frame_2]

    correspondences = build_correspondences(corners1, corners2)

    essential_mat, mask_essential = cv2.findEssentialMat(
        correspondences[1],
        correspondences[2],
        intrinsic_mat,
        method=cv2.RANSAC,
        threshold=1.0)

    _, mask_homography = cv2.findHomography(correspondences[1], correspondences[2], method=cv2.RANSAC)

    essential_inliers, homography_inliers = mask_essential.flatten().sum(), mask_homography.flatten().sum()

    if homography_inliers > essential_inliers * 0.5:
        return None, 0

    correspondences = _remove_correspondences_with_ids(correspondences, np.argwhere(mask_essential == 0))

    R1, R2, t = cv2.decomposeEssentialMat(essential_mat)

    candidates = [Pose(R1.T, R1.T @ t), Pose(R1.T, R1.T @ (-t)), Pose(R2.T, R2.T @ t), Pose(R2.T, R2.T @ (-t))]

    best_pose_score, best_pose = 0, None

    triangulation_parameters = TriangulationParameters(1, 2, .1)
    for pose in candidates:
        points, _, _ = triangulate_correspondences(correspondences,
                                                   eye3x4(),
                                                   pose_to_view_mat3x4(pose),
                                                   intrinsic_mat,
                                                   triangulation_parameters)
        if len(points) > best_pose_score:
            best_pose_score = len(points)
            best_pose = pose

    return best_pose, best_pose_score


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frames_n = len(rgb_sequence)

    if known_view_1 is None or known_view_2 is None:
        (pose1_idx, pose2_idx), pose1, pose2 = find_best_start_poses(frames_n, corner_storage, intrinsic_mat)
        known_view_1, known_view_2 = (pose1_idx, pose1), (pose2_idx, pose2)

    view_mats, point_cloud_builder = CameraTracker(intrinsic_mat, corner_storage, known_view_1, known_view_2,
                                                   frames_n).track()

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
