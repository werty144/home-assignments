import numpy as np
import cv2

from _camtrack import (
    PointCloudBuilder,
    triangulate_correspondences,
    TriangulationParameters,
    build_correspondences,
    pose_to_view_mat3x4,
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4,
    check_baseline,
    to_camera_center
)

MAX_REPROJECTION_ERROR = 4


def camera_pos(frame_n, corner_storage, pcb, intrinsic_mat, outliers):
    corners = corner_storage[frame_n]
    _, points_ids, corners_ids = np.intersect1d(pcb.ids, corners.ids, assume_unique=True, return_indices=True)

    common_points = pcb.points[points_ids]
    common_corner_ids = pcb.ids[points_ids]
    common_corners = corners.points[corners_ids]
    bool_mask = np.full_like(common_points, False, dtype=np.bool)
    for common_id, mask_elem in zip(common_corner_ids[:, 0], bool_mask):
        if common_id not in outliers:
            mask_elem[:] = True

    common_points = common_points[bool_mask].reshape(-1, 3)
    common_corner_ids = common_corner_ids[bool_mask[:, :1]].reshape(-1, 1)
    common_corners = common_corners[bool_mask[:, :2]].reshape(-1, 2)

    _, r_vec, t_vec, inliers = cv2.solvePnPRansac(common_points,
                                                  common_corners,
                                                  intrinsic_mat,
                                                  None,
                                                  reprojectionError=MAX_REPROJECTION_ERROR,
                                                  flags=cv2.SOLVEPNP_EPNP)

    inlier_points = common_points[inliers]
    inlier_corners = common_corners[inliers]
    outlier_ids = np.setdiff1d(common_corner_ids, common_corner_ids[inliers], assume_unique=True)
    outliers.update(outlier_ids)

    print(f'{len(points_ids)} points in cloud')
    print(f'{len(inlier_corners)} inliers')

    _, r_vec, t_vec, inliers = cv2.solvePnPRansac(inlier_points, inlier_corners, intrinsic_mat, None,
                                                  r_vec, t_vec, useExtrinsicGuess=True)

    return rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)


def points_to_cloud(frame1_n,
                    frame2_n,
                    corner_storage,
                    tracked_poses,
                    intrinsic_mat,
                    pcb):
    corners1 = corner_storage[frame1_n]
    corners2 = corner_storage[frame2_n]
    b_cor = build_correspondences(corners1, corners2, ids_to_remove=pcb.ids)
    if len(b_cor.ids) == 0:
        return

    view_1 = tracked_poses[frame1_n]
    view_2 = tracked_poses[frame2_n]
    triangl_params = TriangulationParameters(MAX_REPROJECTION_ERROR, 1, 0)
    points, triangl_ids, med_cos = triangulate_correspondences(b_cor, view_1, view_2, intrinsic_mat, triangl_params)

    print(f'Added {len(points)} points')
    pcb.add_points(triangl_ids, points)


def track(intrinsic_mat, corner_storage, known_view_1, known_view_2):
    start_frame1 = min(known_view_1[0], known_view_2[0])
    start_frame2 = max(known_view_1[0], known_view_2[0])

    view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
    tracked_positions = [eye3x4()] * len(corner_storage)
    tracked_positions[known_view_1[0]] = view_mat_1
    tracked_positions[known_view_2[0]] = view_mat_2
    start_baseline = np.linalg.norm(to_camera_center(view_mat_2) - to_camera_center(view_mat_1))

    pcb = PointCloudBuilder()

    outliers = set()

    points_to_cloud(known_view_1[0], known_view_2[0], corner_storage, tracked_positions, intrinsic_mat, pcb)

    cur_frame = start_frame1 + 1
    frame_n = len(corner_storage)
    for _ in range(2, frame_n):
        cur_frame += cur_frame == start_frame2

        print(f'Current frame: {cur_frame}')
        tracked_positions[cur_frame] = camera_pos(cur_frame, corner_storage, pcb, intrinsic_mat, outliers)
        frame_diff = 5
        pairs_cnt = 0
        while pairs_cnt < 5:
            prev_frame = cur_frame - frame_diff if cur_frame > start_frame1 else cur_frame + frame_diff
            frame_diff += 1
            if prev_frame < frame_n and (start_frame1 <= prev_frame or cur_frame < start_frame1):
                if check_baseline(tracked_positions[prev_frame], tracked_positions[cur_frame], start_baseline / 6):
                    points_to_cloud(prev_frame, cur_frame, corner_storage, tracked_positions, intrinsic_mat, pcb)
                    pairs_cnt += 1
            else:
                break
        cur_frame = cur_frame + 1 if cur_frame > start_frame1 else cur_frame - 1

    return tracked_positions, pcb
