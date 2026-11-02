import cv2
import numpy as np


def mask_preprocessing(mask, visualize=False, anchor_ratio=0.25):
    """Fit an anchored glottal midline and extract bilateral distance features."""
    if mask is None:
        empty_vec = np.zeros(8, dtype=np.float32)
        if visualize:
            return None, _get_empty_features(), empty_vec, None
        return None, _get_empty_features(), empty_vec

    h, w = mask.shape
    vis_img = None
    if visualize:
        vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    mask_128 = (mask == 128)
    mask_255 = (mask == 255)
    
    rows_128 = np.where(mask_128.any(axis=1))[0]
    rows_255 = np.where(mask_255.any(axis=1))[0]
    
    feature_vector = np.zeros(8, dtype=np.float32)
    
    if len(rows_128) == 0 or len(rows_255) == 0:
        if visualize:
            return mask.copy(), _get_empty_features(), feature_vector, vis_img
        return mask.copy(), _get_empty_features(), feature_vector

    min_y = max(rows_128.min(), rows_255.min())
    max_y = min(rows_128.max(), rows_255.max())

    points_128_inner = [] 
    points_255_inner = [] 
    midpoints = []        

    for y in range(min_y, max_y + 1):
        row_l = mask_128[y, :]
        row_r = mask_255[y, :]
        
        if np.any(row_l) and np.any(row_r):
            x_l = np.where(row_l)[0][-1] 
            x_r = np.where(row_r)[0][0]
            
            if x_l < x_r:
                points_128_inner.append((x_l, y))
                points_255_inner.append((x_r, y))
                mid_x = (x_l + x_r) / 2.0
                midpoints.append((mid_x, y))

    if len(midpoints) < 5:
        if visualize:
            return mask.copy(), _get_empty_features(), feature_vector, vis_img
        return mask.copy(), _get_empty_features(), feature_vector

    # Fit the midline using anterior/posterior anchor points.
    n_pts = len(midpoints)
    n_anchor = int(n_pts * anchor_ratio)
    
    if n_anchor > 2:
        anchors = midpoints[:n_anchor] + midpoints[-n_anchor:]
        pts_for_fit = np.array(anchors, dtype=np.float32)
        fit_color = (0, 255, 0)
    else:
        pts_for_fit = np.array(midpoints, dtype=np.float32)
        fit_color = (0, 255, 255)
    
    line_params = cv2.fitLine(pts_for_fit, cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line_params[0], line_params[1], line_params[2], line_params[3]

    if visualize and vis_img is not None:
        for (px, py) in points_128_inner + points_255_inner:
            vis_img[py, px] = [0, 0, 255]
        if n_anchor > 2:
            for (mx, my) in midpoints[:n_anchor] + midpoints[-n_anchor:]:
                 cv2.circle(vis_img, (int(mx), int(my)), 1, (255, 0, 0), -1)
        if abs(vy) > 1e-6:
            t_top = (0 - y0) / vy
            x_top = int(x0 + t_top * vx)
            t_bottom = (h - y0) / vy
            x_bottom = int(x0 + t_bottom * vx)
            cv2.line(vis_img, (x_top, 0), (x_bottom, h), fit_color, 2)
        else:
            cv2.line(vis_img, (0, int(y0)), (w, int(y0)), fit_color, 2)

    # [L_mean, L_var, L_std, R_mean, R_var, R_std, L_exist, R_exist]
    features = _get_empty_features()
    
    def get_perp_distance(pts, vx, vy, x0, y0):
        if not pts: return []
        dists = []
        for (px, py) in pts:
            vec_x = px - x0
            vec_y = py - y0
            cross_prod = vec_x * vy - vec_y * vx
            dists.append(abs(cross_prod[0])) 
        return dists

    dists_128 = get_perp_distance(points_128_inner, vx, vy, x0, y0)
    if dists_128:
        features[128]['raw'] = dists_128
        features[128]['mean'] = np.mean(dists_128)
        features[128]['var'] = np.var(dists_128)
        features[128]['std'] = np.std(dists_128)
        feature_vector[0] = features[128]['mean']
        feature_vector[1] = features[128]['var']
        feature_vector[2] = features[128]['std']
        feature_vector[6] = 1.0 # L_exist

    dists_255 = get_perp_distance(points_255_inner, vx, vy, x0, y0)
    if dists_255:
        features[255]['raw'] = dists_255
        features[255]['mean'] = np.mean(dists_255)
        features[255]['var'] = np.var(dists_255)
        features[255]['std'] = np.std(dists_255)
        feature_vector[3] = features[255]['mean']
        feature_vector[4] = features[255]['var']
        feature_vector[5] = features[255]['std']
        feature_vector[7] = 1.0 # R_exist

    if visualize:
        return mask, features, feature_vector, vis_img
    return mask, features, feature_vector

def _get_empty_features():
    return {
        128: {'raw': [], 'mean': 0.0, 'var': 0.0, 'std': 0.0},
        255: {'raw': [], 'mean': 0.0, 'var': 0.0, 'std': 0.0}
    }
