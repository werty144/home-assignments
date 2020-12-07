import numpy as np

from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    TriangulationParameters,
    Correspondences,
    compute_reprojection_errors,
    eye3x4
)
import cv2


class PointInfo:
    """
    Class for storing info about point from the cloud.
    It's position and number of inliers which can show
    how good this position is.
    """

    def __init__(self, pos, inliers):
        self.pos = pos
        self.inliers = inliers


class CameraInfo:
    """
    Class for storing info about found camera position.
    It's position and number of inliers which can show
    how good this position is.
    """

    def __init__(self, pos, inliers):
        self.pos = pos
        self.inliers = inliers


class CameraTracker:
    MAX_REPROJ_ERR = 1.5

    def _match_corner_with_frames(self):
        for frame in range(self.num_of_frames):
            for index, corner_id in enumerate(self.corner_storage[frame].ids.flatten()):
                if corner_id not in self.corner_pos_in_frames.keys():
                    self.corner_pos_in_frames[corner_id] = []
                self.corner_pos_in_frames[corner_id].append((frame, index))

    def _set_initial_poses(self, known_view_1, known_view_2):
        view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
        view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
        self.tracked_poses[known_view_1[0]] = CameraInfo(view_mat_1, float('inf'))
        self.tracked_poses[known_view_2[0]] = CameraInfo(view_mat_2, float('inf'))

    def _init_point_cloud(self, known_view_1, known_view_2):
        init_cloud_pts, init_ids = self._triangulate(known_view_1[0], known_view_2[0])
        print(f'Init point cloud: added {len(init_cloud_pts)} points.')
        self._update_point_cloud(init_cloud_pts, init_ids, 2 * np.ones_like(init_ids))

    def __init__(self, intrinsic_mat, corner_storage, known_view_1, known_view_2, num_of_frames):
        self.intrinsic_mat = intrinsic_mat
        self.corner_storage = corner_storage
        self.num_of_frames = num_of_frames
        self.point_cloud = {}
        self.corner_pos_in_frames = {}
        self._match_corner_with_frames()
        self.tracked_poses = [None] * self.num_of_frames
        self._set_initial_poses(known_view_1, known_view_2)
        self._init_point_cloud(known_view_1, known_view_2)

    def _update_point_cloud(self, pts, ids, inliers):
        num_of_updated_pts = 0
        for pt_id, pt, inl in zip(ids, pts, inliers):
            if pt_id not in self.point_cloud.keys() or inl >= self.point_cloud[pt_id].inliers:
                num_of_updated_pts += 1
                self.point_cloud[pt_id] = PointInfo(pt, inl)
        return num_of_updated_pts

    def _get_pos(self, frame_number):
        corners = self.corner_storage[frame_number]
        cur_corners, cur_cloud_pts = [], []
        for i, corner in zip(corners.ids.flatten(), corners.points):
            if i in self.point_cloud.keys():
                cur_corners.append(corner)
                cur_cloud_pts.append(self.point_cloud[i].pos)
        cur_corners, cur_cloud_pts = np.array(cur_corners), np.array(cur_cloud_pts)
        if len(cur_cloud_pts) < 4:
            return None  # Not enough points for ransac

        is_success, r_vec, t_vec, inliers = cv2.solvePnPRansac(cur_cloud_pts,
                                                               cur_corners,
                                                               self.intrinsic_mat,
                                                               None,
                                                               flags=cv2.SOLVEPNP_EPNP)
        if not is_success:
            return None

        _, r_vec, t_vec = cv2.solvePnP(cur_cloud_pts[inliers],
                                       cur_corners[inliers],
                                       self.intrinsic_mat,
                                       distCoeffs=None,
                                       flags=cv2.SOLVEPNP_ITERATIVE,
                                       useExtrinsicGuess=True,
                                       rvec=r_vec,
                                       tvec=t_vec
                                       )

        pos = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        inliers_n = len(np.array(inliers).flatten())
        return CameraInfo(pos, inliers_n)

    def _triangulate(self, frame_num_1, frame_num_2):
        corners_1 = self.corner_storage[frame_num_1]
        corners_2 = self.corner_storage[frame_num_2]
        corresps = build_correspondences(corners_1,
                                         corners_2,
                                         ids_to_remove=np.array(list(map(int, self.point_cloud.keys())), dtype=int))

        view_1 = self.tracked_poses[frame_num_1].pos
        view_2 = self.tracked_poses[frame_num_2].pos
        triangulation_params = TriangulationParameters(1, 1, 0)
        pts_3d, triangulated_ids, med_cos = triangulate_correspondences(corresps,
                                                                        view_1,
                                                                        view_2,
                                                                        self.intrinsic_mat,
                                                                        triangulation_params)

        return pts_3d, triangulated_ids

    def _retriangulate(self, point_id, max_pairs=5):
        frames, corners, poses = [], [], []
        for frame, index_on_frame in self.corner_pos_in_frames[point_id]:
            if self.tracked_poses[frame] is not None:
                frames.append(frame)
                corners.append(self.corner_storage[frame].points[index_on_frame])
                poses.append(self.tracked_poses[frame].pos)

        if len(frames) < 2:
            return None
        if len(frames) == 2:
            cloud_pts, _ = self._triangulate(frames[0], frames[1])
            if len(cloud_pts) == 0:
                return None
            return cloud_pts[0], 2

        frames, corners, poses = np.array(frames), np.array(corners), np.array(poses)

        best_pos, best_inliers = None, None
        for _ in range(max_pairs):
            frame_1, frame_2 = np.random.choice(len(frames), 2, replace=False)
            corner_pos_1, corner_pos_2 = corners[frame_1], corners[frame_2]
            cloud_pts, _, _ = triangulate_correspondences(
                Correspondences(np.zeros(1), np.array([corner_pos_1]), np.array([corner_pos_2])),
                poses[frame_1], poses[frame_2], self.intrinsic_mat,
                TriangulationParameters(self.MAX_REPROJ_ERR, 2.5, 0.0))
            if len(cloud_pts) == 0:
                continue

            inliers = 0
            for frame, corner in zip(frames, corners):
                repr_errors = compute_reprojection_errors(
                    cloud_pts,
                    np.array([corner]),
                    self.intrinsic_mat @ self.tracked_poses[frame].pos
                )
                inliers += np.sum(repr_errors <= self.MAX_REPROJ_ERR)

            if best_pos is None or best_inliers < inliers:
                best_pos = cloud_pts[0]
                best_inliers = inliers

        if best_pos is None:
            return None
        return best_pos, best_inliers

    def _retriangulate_3d_points(self, frame, max_points=700):
        points = [i for i in self.corner_storage[frame].ids.flatten()]
        np.random.shuffle(points)
        points = points[:max_points]

        retr_pts, retr_ids, retr_inliers = [], [], []
        for i, retr_result in zip(points, map(self._retriangulate, points)):
            if retr_result is not None:
                cloud_pt, inliers = retr_result
                retr_pts.append(cloud_pt)
                retr_ids.append(i)
                retr_inliers.append(inliers)

        updated_points_n = self._update_point_cloud(retr_pts, retr_ids, retr_inliers)
        print(f'Updated points in the cloud: {updated_points_n}')

    def _update_camera_poses(self):
        solved_frames = [i for i in range(self.num_of_frames) if self.tracked_poses[i] is not None]
        updated_poses = 0
        for i, cam_info in zip(solved_frames, map(self._get_pos, solved_frames)):
            if cam_info is not None:
                if cam_info.inliers >= self.tracked_poses[i].inliers:
                    updated_poses += 1
                    self.tracked_poses[i] = cam_info
        print(f'Updated camera positions: {updated_poses}')

    def track(self):
        step_num = 1
        num_of_defined_poses = np.sum([track_pos_info is not None for track_pos_info in self.tracked_poses])
        while num_of_defined_poses != self.num_of_frames:
            unsolved_frames = [i for i in range(self.num_of_frames) if self.tracked_poses[i] is None]

            new_cams_info = []
            for frame, cam_info in zip(unsolved_frames, map(self._get_pos, unsolved_frames)):
                if cam_info is not None:
                    new_cams_info.append((frame, cam_info))

            if len(new_cams_info) == 0:
                print(
                    f'Can not get more camera positions, '
                    f'{self.num_of_frames - num_of_defined_poses}'
                    f' frames left without defined camera position')
                self.tracked_poses = [CameraInfo(view_mat3x4_to_pose(eye3x4()), 0) if pos is None
                                      else pos for pos in self.tracked_poses]
                break

            best_frame = None
            best_new_cam_info = None

            for frame, cam_info in new_cams_info:
                if best_new_cam_info is None or best_new_cam_info.inliers < cam_info.inliers:
                    best_new_cam_info = cam_info
                    best_frame = frame

            print('Added camera position for frame ', best_frame)
            print('Number of inliers: ', best_new_cam_info.inliers)

            self.tracked_poses[best_frame] = best_new_cam_info

            self._retriangulate_3d_points(best_frame)

            if step_num % 4 == 0:
                self._update_camera_poses()
            step_num += 1
            num_of_defined_poses = np.sum([tracked_pos_info is not None for tracked_pos_info in self.tracked_poses])
            print(
                f'{num_of_defined_poses}/{self.num_of_frames} camera positions found, {len(self.point_cloud)} points in cloud')

        self._update_camera_poses()

        ids, cloud_points = [], []
        for pt_id, clout_pt_info in self.point_cloud.items():
            ids.append(pt_id)
            cloud_points.append(clout_pt_info.pos)
        return list(map(lambda tracked_pos_info: tracked_pos_info.pos, self.tracked_poses)), \
               PointCloudBuilder(np.array(ids), np.array(cloud_points))
