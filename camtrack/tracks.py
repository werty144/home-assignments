import numpy as np
import cv2

feature_params = dict(qualityLevel=0.01,
                      minDistance=5,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))


def resize_to_lvl(img, lvl):
    if lvl == 1:
        return img
    return cv2.resize(img, (int(img.shape[1] / lvl), int(img.shape[0] / lvl)))


def corners_n_on_lvl(lvl):
    return int(3000 / lvl ** 2)


def get_corners_on_lvl(img, lvl):
    max_corners = corners_n_on_lvl(lvl)
    resized = resize_to_lvl(img, lvl)
    corners = cv2.goodFeaturesToTrack(resized, mask=None, **feature_params, maxCorners=max_corners)
    return (corners * lvl).astype(np.float32)


def get_corners(img, levels_n=2):
    corners = []
    radii = np.array([])
    for lvl in range(1, levels_n + 1):
        corners_on_lvl = get_corners_on_lvl(img, lvl)
        corners.append(corners_on_lvl)
        radii = np.concatenate((radii, np.full(corners_on_lvl.shape[0], lvl * 5)))
    return np.concatenate(corners).reshape((-1, 1, 2)), radii


def new_corners(old_img, new_img, prev_corn, prev_ids, prev_radii, max_level):
    new_corners, st, _ = cv2.calcOpticalFlowPyrLK(old_img, new_img, prev_corn, None, maxLevel=max_level, **lk_params)
    old_corners_reversed, _, _ = cv2.calcOpticalFlowPyrLK(new_img, old_img, new_corners, None, **lk_params)

    d = abs(prev_corn - old_corners_reversed).reshape(-1, 2).max(-1)
    good_new_corners = new_corners[d < 1]
    return good_new_corners, prev_ids[d < 1], prev_radii[d < 1]


def restore_loss(img, lvl, good_new, spare_id, loss):
    resized = resize_to_lvl(img, lvl) if lvl > 1 else img
    good_new = good_new / lvl
    additional_corners = cv2.goodFeaturesToTrack(resized, mask=None, **feature_params, maxCorners=corners_n_on_lvl(lvl))
    result = []
    for corner in additional_corners.reshape((-1, 2)):
        min_dist = np.min(np.sum((good_new - corner) ** 2, axis=2))
        if min_dist >= feature_params['minDistance']:
            result.append(corner)
    additional_corners = np.array(result[:loss]).reshape((-1, 1, 2))
    return (additional_corners * lvl).astype(np.float32), \
            np.arange(spare_id, spare_id + additional_corners.shape[0]), \
            np.full(additional_corners.shape[0], lvl * 5)


def get_corners_video(frame_sequence, levels_n=2):
    prev_frame = (frame_sequence[0] * 255).astype(np.uint8)
    prev_corners, cur_radii = get_corners(prev_frame, levels_n)
    cur_ids = np.arange(prev_corners.shape[0])
    ids = [cur_ids]
    spare_id = prev_corners.shape[0]
    corners = [prev_corners]
    radii = [cur_radii]
    for frame in frame_sequence[1:]:
        frame = (frame * 255).astype(np.uint8)
        good_new, cur_ids, cur_radii = new_corners(prev_frame, frame, prev_corners, cur_ids, cur_radii, levels_n)
        loss = sum([corners_n_on_lvl(lvl) for lvl in range(1, levels_n + 1)]) - good_new.shape[0]
        for lvl in range(1, levels_n + 1):
            if loss > 0:
                restored, new_ids, new_radii = restore_loss(frame, lvl, good_new, spare_id, loss)
                good_new = np.concatenate((good_new, restored)).reshape((-1, 1, 2))
                cur_ids = np.concatenate((cur_ids, new_ids))
                spare_id += new_ids.size
                cur_radii = np.concatenate((cur_radii, new_radii))
                loss -= restored.shape[0]

        prev_corners = good_new
        prev_frame = frame.copy()
        corners.append(good_new)
        radii.append(np.array(cur_radii).ravel())
        ids.append(cur_ids)
        assert good_new.shape[0] == cur_ids.size

    return ids, corners, radii