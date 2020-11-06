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
    to_camera_center,
    compute_reprojection_errors,
    Correspondences
)

MAX_REPROJECTION_ERROR = 4


def is_initial_pos(pos):
    return (pos == eye3x4()).all()


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

    if len(common_points) < 4:
        return None
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

    _, r_vec, t_vec = cv2.solvePnP(inlier_points, inlier_corners, intrinsic_mat, None,
                                   r_vec, t_vec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    return r_vec, t_vec, np.array(inliers).size


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

    frame_n = len(corner_storage)
    point_cloud = {}

    corner_frame = {}
    for frame in range(frame_n):
        for index, corner_id in enumerate(corner_storage[frame].ids.flatten()):
            if corner_id not in corner_frame.keys():
                corner_frame[corner_id] = list()
            corner_frame[corner_id].append((frame, index))

    points_to_cloud(known_view_1[0], known_view_2[0], corner_storage, tracked_positions, intrinsic_mat, pcb)

    cur_frame = start_frame1 + 1
    for _ in range(2, frame_n):
        cur_frame += cur_frame == start_frame2
        cur_frame %= frame_n

        cam_pos = camera_pos(cur_frame, corner_storage, pcb, intrinsic_mat, outliers)
        if cam_pos is None:
            cur_frame = cur_frame + 1 if cur_frame > start_frame1 else cur_frame - 1
            continue
        rvec, tvec, _ = cam_pos
        tracked_positions[cur_frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
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

            best_frame = calculate_best_frame(tracked_positions, frame_n, corner_storage, pcb, intrinsic_mat, outliers)
            if best_frame is None:
                continue
            points_to_cloud_with_triangulation(best_frame,
                                               tracked_positions,
                                               intrinsic_mat,
                                               corner_frame,
                                               corner_storage,
                                               point_cloud)

        cur_frame = cur_frame + 1 if cur_frame > start_frame1 else cur_frame - 1

    return tracked_positions, pcb


def calculate_best_frame(tracked_positions, frame_n, corner_storage, pcb, intrinsic_mat, outliers):
    unsolved_frames = [i for i in range(frame_n) if is_initial_pos(tracked_positions[i])]

    new_positions_n_inliers = list()
    for frame, position_n_inliers in zip(unsolved_frames, map(
            lambda frame_ind: camera_pos(frame_ind, corner_storage, pcb, intrinsic_mat, outliers),
            unsolved_frames)):
        if position_n_inliers is not None:
            new_positions_n_inliers.append((frame, position_n_inliers))

    best_frame = None
    best_inliers_n = -1

    for frame, position in new_positions_n_inliers:
        if best_inliers_n < position[2]:
            best_inliers_n = position[2]
            best_frame = frame

    print(f'Best frame is {best_frame}')
    return best_frame


def retriangulate(corner_id, tracked_poses, corner_storage, intrinsic_mat, corner_pos_in_frames):
    frames = list()
    corners = list()
    positions = list()
    for frame, index_on_frame in corner_pos_in_frames[corner_id]:
        if not is_initial_pos(tracked_poses[frame]):
            frames.append(frame)
            corners.append(corner_storage[frame].points[index_on_frame])
            positions.append(tracked_poses[frame])

    indices = np.arange(len(frames))
    np.random.shuffle(indices)
    rand_ind = indices[:10]
    frames, corners, positions = np.array(frames)[rand_ind], np.array(corners)[rand_ind], np.array(positions)[rand_ind]

    best_pos = None
    best_inliers_n = -1
    for _ in range(5):
        if len(frames) < 2:
            break
        frame1, frame2 = np.random.choice(len(frames), 2, replace=False)
        corner1, corner2 = corners[frame1], corners[frame2]
        cur_pts, _, _ = triangulate_correspondences(
            Correspondences(np.zeros(1), np.array([corner1]), np.array([corner2])),
            positions[frame1], positions[frame2], intrinsic_mat,
            TriangulationParameters(MAX_REPROJECTION_ERROR, 2.5, 0.0))
        if len(cur_pts) == 0:
            continue

        inliers_n = 0
        for frame, corner in zip(frames, corners):
            inliers_n += (compute_reprojection_errors(cur_pts, np.array([corner]), intrinsic_mat @ tracked_poses[frame])
                          <= MAX_REPROJECTION_ERROR).sum()

        if best_pos is None or best_inliers_n < inliers_n:
            best_pos = cur_pts[0]
            best_inliers_n = inliers_n

    if best_pos is None:
        return None
    return best_pos, best_inliers_n


def points_to_cloud_with_triangulation(frame,
                                       tracked_poses,
                                       intrinsic_mat,
                                       corner_pos_in_frames,
                                       corner_storage,
                                       point_cloud,
                                       max_retrs=700,
                                       max_iters=100):

    points = [i for i in corner_storage[frame].ids.flatten()]
    np.random.shuffle(points)
    points = points[:max_retrs]

    cur_poitns = list()
    retriangulated_ids = list()
    retriangulated_inliers = list()
    iter_counter = 0
    for ind, retriangulation in zip(points, map(
            lambda corner: retriangulate(corner, tracked_poses, corner_storage, intrinsic_mat, corner_pos_in_frames),
            points)):
        if iter_counter > max_iters:
            break
        iter_counter += 1
        if retriangulation is not None:
            cloud_poitns, inliers = retriangulation
            cur_poitns.append(cloud_poitns)
            retriangulated_ids.append(ind)
            retriangulated_inliers.append(inliers)

    iter_counter = 0
    num_of_updated_pts = 0
    for point_id, point, inlier in zip(retriangulated_ids, cur_poitns, retriangulated_inliers):
        if iter_counter > max_iters:
            break
        iter_counter += 1
        if point_id not in point_cloud.keys() or inlier >= point_cloud[point_id][1]:
            num_of_updated_pts += 1
            point_cloud[point_id] = (point, inlier)
