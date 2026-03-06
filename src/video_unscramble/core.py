import cv2
import numpy as np
from tqdm import tqdm
import cv2
import os
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


INVALID_SCORE = -1e6

def extract_frames(video_path: str) -> List[np.ndarray]:
    """
    Extract all frames from a video.

    Args:
        video_path: Path to the input video file.

    Returns:
        List of frames as NumPy arrays.
    """
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or not isinstance(frame, np.ndarray):
            print(f"[WARNING] Skipping invalid frame at position {frame_count}.")
            frame_count += 1
            continue
        frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

def compute_histogram_features(frames: List[np.ndarray], bins: int = 64, resize: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Compute grayscale histograms for all frames.

    Args:
        frames: List of frames.
        bins: Number of histogram bins.
        resize: Target resize dimensions.

    Returns:
        2D array of histogram features (n_frames x bins).
    """
    features = np.zeros((len(frames), bins), dtype=np.float32)
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resize to speed up histogram computation
        if resize is not None:
            gray = cv2.resize(gray, resize)
        
        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features[idx] = hist
    return features


def compute_frame_embeddings(
    frames: List[np.ndarray],
    bins: int = 32,
    resize: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Build richer frame descriptors for clustering and outlier detection.

    The descriptor mixes coarse appearance, color distribution, edge structure,
    and low-order image statistics. This is still lightweight, but much more
    discriminative than a single grayscale histogram.
    """
    embeddings = []
    for frame in frames:
        resized = cv2.resize(frame, resize) if resize is not None else frame
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        gray_hist = cv2.normalize(
            cv2.calcHist([gray], [0], None, [bins], [0, 256]),
            None,
        ).flatten()
        sat_hist = cv2.normalize(
            cv2.calcHist([hsv], [1], None, [bins], [0, 256]),
            None,
        ).flatten()
        val_hist = cv2.normalize(
            cv2.calcHist([hsv], [2], None, [bins], [0, 256]),
            None,
        ).flatten()

        edges = cv2.Canny(gray, 80, 160)
        edge_hist = cv2.normalize(
            cv2.calcHist([edges], [0], None, [8], [0, 256]),
            None,
        ).flatten()

        stats = np.array(
            [
                float(gray.mean()),
                float(gray.std()),
                float(hsv[..., 0].mean()),
                float(hsv[..., 1].mean()),
                float(hsv[..., 2].mean()),
                float(edges.mean() / 255.0),
                float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            ],
            dtype=np.float32,
        )
        embeddings.append(np.concatenate([gray_hist, sat_hist, val_hist, edge_hist, stats]))

    return np.asarray(embeddings, dtype=np.float32)


def reduce_frame_embeddings(features: np.ndarray, max_components: int = 24) -> np.ndarray:
    """
    Standardize and compress frame descriptors before clustering.
    """
    if features.size == 0:
        return features

    scaled = StandardScaler().fit_transform(features)
    if len(features) < 3:
        return scaled.astype(np.float32)

    n_components = min(max_components, scaled.shape[1], len(features) - 1)
    if n_components < 2:
        return scaled.astype(np.float32)

    reduced = PCA(n_components=n_components, random_state=42).fit_transform(scaled)
    return reduced.astype(np.float32)

def cluster_frames(features: np.ndarray, n_clusters: int, max_iter: int = 300, mini: bool = True) -> np.ndarray:
    """
    Cluster frames using K-Means or MiniBatchKMeans.

    Args:
        features: Feature matrix for frames.
        n_clusters: Number of clusters.
        max_iter: Maximum iterations for K-Means.
        mini: Whether to use MiniBatchKMeans.

    Returns:
        Array of cluster labels for each frame.
    """

    if len(features) == 0:
        return np.array([], dtype=np.int32)

    reduced = reduce_frame_embeddings(features)
    n_clusters = max(1, min(n_clusters, len(reduced)))

    if len(reduced) < n_clusters + 1:
        labels = np.zeros(len(reduced), dtype=np.int32)
    else:
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            reg_covar=1e-5,
            random_state=42,
            max_iter=max_iter,
            n_init=3,
        )
        labels = gmm.fit_predict(reduced)
        probs = gmm.predict_proba(reduced).max(axis=1)
        low_confidence = probs < np.quantile(probs, 0.1)
        labels = labels.astype(np.int32)
        labels[low_confidence] = -1

    non_outliers = labels[labels != -1]
    if non_outliers.size == 0:
        return labels

    dominant_label = int(np.bincount(non_outliers).argmax())
    dominant_mask = labels == dominant_label

    dominant_features = reduced[dominant_mask]
    if len(dominant_features) >= 8:
        contamination = min(0.2, max(0.03, 2.0 / len(dominant_features)))
        detector = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        inlier_flags = detector.fit_predict(dominant_features)
        dominant_indices = np.where(dominant_mask)[0]
        labels[dominant_indices[inlier_flags == -1]] = -1

    return labels.astype(np.int32)

def filter_clusters(frames: List[np.ndarray], labels: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int], Optional[int]]:
    """
    Split frames into inliers and outliers based on dominant cluster.

    Args:
        frames: List of frames.
        labels: Cluster labels for each frame.

    Returns:
        inliers, outliers, inlier indices, outlier indices, dominant cluster label.
    """
    # Identify the dominant cluster (exclude label -1 used for outliers)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    label_counts = {
        int(lab): int(cnt)
        for lab, cnt in zip(unique_labels, counts)
        if lab != -1
    }
    if label_counts:
        dominant_label = max(label_counts, key=label_counts.get)
    else:
        dominant_label = None

    inliers: List[np.ndarray] = []
    outliers: List[np.ndarray] = []
    inlier_indices: List[int] = []
    outlier_indices: List[int] = []
    for idx, (frame, lab) in enumerate(zip(frames, labels)):
        if lab == dominant_label:
            inliers.append(frame)
            inlier_indices.append(idx)
        else:
            outliers.append(frame)
            outlier_indices.append(idx)
    return inliers, outliers, inlier_indices, outlier_indices, dominant_label

def save_frames(frames: List[np.ndarray], indices: List[int], directory: str, prefix: str = "frame") -> None:
    """
    Save selected frames as JPEG images to a directory.

    Args:
        frames: Frames to save.
        indices: Frame indices for filenames.
        directory: Output directory path.
        prefix: Filename prefix.
    """
    os.makedirs(directory, exist_ok=True)
    for frame, idx in zip(frames, indices):
        filename = os.path.join(directory, f"{prefix}_{idx:05d}.jpg")
        success = cv2.imwrite(filename, frame)
        if not success:
            print(f"[WARNING] Could not save frame index {idx} to {filename}")

def compute_feature_matches_SIFT(frames: List[np.ndarray], max_features: int = 500, min_good_matches: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise SIFT feature matches and motion between frames.

    Args:
        frames: List of frames.
        max_features: Max keypoints per frame.
        min_good_matches: Minimum matches to retain a pair.

    Returns:
        match_matrix, motion_matrix.
    """
    RATIO_THRESH = 0.75 
    MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)

    n = len(frames)
    sift = cv2.SIFT_create( nfeatures=500,
        nOctaveLayers=4,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6)

    keypoints: List[list] = []
    descriptors: List[Optional[np.ndarray]] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        if not kp or desc is None:
            keypoints.append([])
            descriptors.append(None)
            continue

        if len(kp) > max_features:
            responses = np.array([k.response for k in kp])
            idx = np.argpartition(-responses, max_features - 1)[:max_features]
            idx = np.sort(idx)
            kp = [kp[i] for i in idx]
            desc = desc[idx]

        keypoints.append(kp)
        descriptors.append(desc)

    match_matrix = np.zeros((n, n), dtype=np.float32)
    motion_matrix = np.full((n, n), np.nan, dtype=np.float32)

    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    def match_pair(i: int, j: int):
        des1, des2 = descriptors[i], descriptors[j]
        if des1 is None or des2 is None:
            return i, j, 0.0, np.nan
        kp1, kp2 = keypoints[i], keypoints[j]

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw = matcher.knnMatch(des1, des2, k=2)
        # Lowe ratio
        good = [m for pair in raw if len(pair)==2 for m,n in [pair] if m.distance < RATIO_THRESH * n.distance]
        if len(good) < min_good_matches:
            return i, j, 0.0, np.nan

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        med = float(np.median(np.linalg.norm(pts1 - pts2, axis=1)))
        return i, j, float(len(good)), med

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(match_pair, i, j) for i, j in pairs]
        for fut in as_completed(futures):
            i, j, cnt, med = fut.result()
            match_matrix[i, j] = match_matrix[j, i] = cnt
            motion_matrix[i, j] = motion_matrix[j, i] = med

    return match_matrix, motion_matrix

def compute_feature_matches_AKAZE(frames: List[np.ndarray], max_features: int = 500, min_good_matches: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute AKAZE matches and motion between all frame pairs.

    Args:
        frames: List of frames.
        max_features: Max keypoints per frame.
        min_good_matches: Minimum matches to retain a pair.

    Returns:
        match_matrix, motion_matrix.
    """
    
    RATIO_THRESH = 0.75     # Lowe ratio for KNN
    MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)

    n = len(frames)
    akaze = cv2.AKAZE_create(nOctaves=4, nOctaveLayers=4)

    keypoints: List[list] = []
    descriptors: List[Optional[np.ndarray]] = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = akaze.detectAndCompute(gray, None)
        if kp is None or desc is None or len(kp) == 0:
            keypoints.append([])
            descriptors.append(None)
            continue

        if len(kp) > max_features:
            responses = np.array([k.response for k in kp])
            keep_idx = np.argpartition(-responses, max_features - 1)[:max_features]
            keep_idx = np.sort(keep_idx)
            kp = [kp[i] for i in keep_idx]
            desc = desc[keep_idx]

        keypoints.append(kp)
        descriptors.append(desc)

    match_matrix = np.zeros((n, n), dtype=np.float32)
    motion_matrix = np.full((n, n), np.nan, dtype=np.float32)

    # Pair list (upper triangle)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def match_pair(i: int, j: int):
        des1, des2 = descriptors[i], descriptors[j]
        if des1 is None or des2 is None:
            return i, j, 0.0, np.nan

        kp1, kp2 = keypoints[i], keypoints[j]

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = matcher.knnMatch(des1, des2, k=2)
        
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < RATIO_THRESH * n.distance]
        if len(good) < min_good_matches:
            return i, j, 0.0, np.nan
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        med = float(np.median(np.linalg.norm(pts1 - pts2, axis=1)))
        return i, j, float(len(good)), med

    if len(pairs) > 0 and MAX_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(match_pair, i, j) for (i, j) in pairs]
            for fut in as_completed(futures):
                i, j, cnt, med = fut.result()
                match_matrix[i, j] = match_matrix[j, i] = cnt
                motion_matrix[i, j] = motion_matrix[j, i] = med
    else:
        for (i, j) in pairs:
            i, j, cnt, med = match_pair(i, j)
            match_matrix[i, j] = match_matrix[j, i] = cnt
            motion_matrix[i, j] = motion_matrix[j, i] = med

    return match_matrix, motion_matrix

def get_resnet50_spatial_extractor():
    """
    Load pretrained ResNet50 and return feature extractor.

    Returns:
        Feature extractor model, preprocessing transforms.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights).eval()
    # Extract feature map from layer3
    return create_feature_extractor(model, {'layer3': 'feat_map'}), weights

def match_resnet_spatial_pair(feat0: torch.Tensor, feat1: torch.Tensor, ratio_thresh: float = 0.75) -> Tuple[int, float]:
    """
    Match spatial features from ResNet and compute displacement.

    Args:
        feat0: Feature map from frame A.
        feat1: Feature map from frame B.
        ratio_thresh: Lowe’s ratio threshold.

    Returns:
        Match count, median pixel displacement.
    """

    C, H, W = feat0.shape
    
    desc0 = feat0.reshape(C, -1).T
    desc1 = feat1.reshape(C, -1).T
    
    desc0 = torch.nn.functional.normalize(desc0, dim=1)
    desc1 = torch.nn.functional.normalize(desc1, dim=1)

    # Compute pairwise L2 distances
    dists = torch.cdist(desc0, desc1, p=2).cpu().numpy()
    
    # Nearest two neighbors
    nn_idx = np.argsort(dists, axis=1)[:, :2]
    nn_d  = np.take_along_axis(dists, nn_idx, axis=1)
    # Apply Lowe's ratio test
    good0 = np.where(nn_d[:,0] < ratio_thresh * nn_d[:,1])[0]

    # Mutual consistency
    rev_nn = np.argsort(dists, axis=0)[:2, :]
    matches = []
    for i in good0:
        j = nn_idx[i, 0]
        if i in rev_nn[:, j]:
            matches.append((i, j))

    match_count = len(matches)
    if match_count == 0:
        return 0, float('nan')

    # Compute pixel displacement in feature grid
    coords0 = np.array([(i//W, i%W) for i, _ in matches])
    coords1 = np.array([(j//W, j%W) for _, j in matches])
    disp = np.linalg.norm(coords0 - coords1, axis=1)
    return match_count, float(np.median(disp))


def compute_spatial_saliency_weights(feats: List[torch.Tensor], keep_ratio: float = 0.35) -> np.ndarray:
    """
    Estimate which spatial locations are informative across the frame set.

    Static background regions are common failure points for scene-level matching.
    This function keeps the spatial cells whose feature activations vary the most
    across frames, which tends to emphasize moving or semantically changing
    regions over static walls, logos, and borders.
    """
    if not feats:
        return np.array([], dtype=np.float32)

    stacked = torch.stack(feats, dim=0)
    location_variance = stacked.var(dim=0, unbiased=False).mean(dim=0).cpu().numpy()
    flat = location_variance.reshape(-1)
    if flat.size == 0:
        return flat.astype(np.float32)

    keep = max(8, int(flat.size * keep_ratio))
    if keep >= flat.size:
        mask = np.ones_like(flat, dtype=np.float32)
    else:
        top_idx = np.argpartition(flat, -keep)[-keep:]
        mask = np.zeros_like(flat, dtype=np.float32)
        mask[top_idx] = 1.0

    selected = flat[mask > 0]
    if selected.size:
        lo = float(selected.min())
        hi = float(selected.max())
        if hi > lo:
            mask[mask > 0] = 0.25 + 0.75 * ((selected - lo) / (hi - lo))
        else:
            mask[mask > 0] = 1.0
    return mask.astype(np.float32)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute a weighted median for non-negative weights.
    """
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = weights.sum()
    if total <= 0:
        return float(np.median(values))
    cdf = np.cumsum(weights)
    idx = np.searchsorted(cdf, 0.5 * total, side="left")
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def match_resnet_spatial_pair_weighted(
    feat0: torch.Tensor,
    feat1: torch.Tensor,
    spatial_weights: np.ndarray,
    ratio_thresh: float = 0.75,
) -> Tuple[float, float]:
    """
    Match ResNet spatial features while suppressing static background regions.

    Returns a weighted match score and a weighted median displacement.
    """
    c, h, w = feat0.shape
    desc0 = feat0.reshape(c, -1).T
    desc1 = feat1.reshape(c, -1).T
    active = spatial_weights > 0
    if active.sum() < 2:
        return match_resnet_spatial_pair(feat0, feat1, ratio_thresh)

    desc0 = desc0[active]
    desc1 = desc1[active]
    weights = spatial_weights[active]

    desc0 = torch.nn.functional.normalize(desc0, dim=1)
    desc1 = torch.nn.functional.normalize(desc1, dim=1)
    dists = torch.cdist(desc0, desc1, p=2).cpu().numpy()
    if dists.shape[1] < 2:
        return 0.0, float("nan")

    nn_idx = np.argsort(dists, axis=1)[:, :2]
    nn_d = np.take_along_axis(dists, nn_idx, axis=1)
    good0 = np.where(nn_d[:, 0] < ratio_thresh * nn_d[:, 1])[0]

    rev_nn = np.argsort(dists, axis=0)[:2, :]
    matches = []
    for i in good0:
        j = nn_idx[i, 0]
        if i in rev_nn[:, j]:
            matches.append((i, j))

    if not matches:
        return 0.0, float("nan")

    active_indices = np.flatnonzero(active)
    coords0 = np.array([(active_indices[i] // w, active_indices[i] % w) for i, _ in matches], dtype=np.float32)
    coords1 = np.array([(active_indices[j] // w, active_indices[j] % w) for _, j in matches], dtype=np.float32)
    disp = np.linalg.norm(coords0 - coords1, axis=1)
    match_weights = np.array([weights[i] for i, _ in matches], dtype=np.float32)
    score = float(match_weights.sum())
    med = weighted_median(disp, match_weights)
    return score, med


def compute_feature_matches_ResNet_spatial(frames: List[np.ndarray], ratio_thresh: float = 0.75, max_workers: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ResNet-based feature matches between all frame pairs.

    Args:
        frames: List of frames.
        ratio_thresh: Ratio threshold for matching.
        max_workers: Thread pool size.

    Returns:
        match_matrix, motion_matrix.
    """
    extractor, weights = get_resnet50_spatial_extractor()

    # Precompute feature maps
    feats = []
    preprocess = weights.transforms()
    for img in tqdm(frames, desc='Extracting spatial features'):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inp = preprocess(pil).unsqueeze(0).cpu()
        with torch.no_grad():
            feat_map = extractor(inp)['feat_map'][0].cpu()
        feats.append(feat_map)

    spatial_weights = compute_spatial_saliency_weights(feats)

    n = len(frames)
    match_mat = np.zeros((n, n), dtype=np.float32)
    motion_mat = np.full((n, n), np.nan, dtype=np.float32)

    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) - 1)

    def worker(i: int, j: int):
        cnt, med = match_resnet_spatial_pair_weighted(feats[i], feats[j], spatial_weights, ratio_thresh)
        return i, j, cnt, med

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i, j) for i, j in pairs]
        for fut in as_completed(futures):
            i, j, cnt, med = fut.result()
            match_mat[i, j] = match_mat[j, i] = cnt
            motion_mat[i, j] = motion_mat[j, i] = med

    return match_mat, motion_mat

def combine_global_local(global_scores: np.ndarray, local_counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Fuse local and global similarity matrices using a weight.

    Args:
        global_scores: Global similarity matrix.
        local_counts: Local match count matrix.
        alpha: Weight for global features.

    Returns:
        Combined similarity matrix.
    """

    global_norm = normalize_similarity_matrix(global_scores)
    local_norm = normalize_similarity_matrix(local_counts)
    return alpha * global_norm + (1.0 - alpha) * local_norm


def normalize_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Robustly normalize a similarity matrix to [0, 1].

    Args:
        matrix: Raw similarity matrix.

    Returns:
        Normalized matrix with invalid values mapped to 0.
    """
    arr = np.asarray(matrix, dtype=np.float32).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)

    values = arr[finite]
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        hi = float(values.max())
        lo = float(values.min())
    if hi <= lo:
        out = np.zeros_like(arr, dtype=np.float32)
        out[finite] = 1.0 if hi > 0 else 0.0
        return out

    out = np.zeros_like(arr, dtype=np.float32)
    out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def normalize_motion_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Robustly normalize a motion matrix to [0, 1].

    Lower motion is better, but this function only rescales values.
    Invalid values are treated as maximum motion.
    """
    arr = np.asarray(matrix, dtype=np.float32).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.ones_like(arr, dtype=np.float32)

    values = arr[finite]
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        hi = float(values.max())
        lo = float(values.min())
    out = np.ones_like(arr, dtype=np.float32)
    if hi <= lo:
        out[finite] = 0.0
        return out

    out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def build_score_matrix(match_matrix: np.ndarray, motion_matrix: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Build a robust transition score matrix from matches and motion.

    Args:
        match_matrix: Pairwise similarity counts/scores.
        motion_matrix: Pairwise displacement estimates.
        alpha: Motion penalty weight.

    Returns:
        Score matrix where larger values indicate more likely adjacency.
    """
    match_norm = normalize_similarity_matrix(match_matrix)
    motion_norm = normalize_motion_matrix(motion_matrix)
    score_matrix = match_norm - alpha * motion_norm
    np.fill_diagonal(score_matrix, INVALID_SCORE)
    return score_matrix.astype(np.float32)


def select_start_candidates(score_matrix: np.ndarray, max_candidates: int = 8) -> List[int]:
    """
    Choose plausible start frames for greedy sequencing.

    Endpoints in a path usually have lower overall connectivity than interior
    frames, so we seed the search from the weakest-connected frames instead of
    relying on a symmetric matrix tie.
    """
    if score_matrix.shape[0] == 0:
        return []

    finite = score_matrix > INVALID_SCORE / 10
    masked = np.where(finite, score_matrix, np.nan)
    connectivity = np.nanmean(masked, axis=1)
    connectivity = np.where(np.isfinite(connectivity), connectivity, np.inf)
    order = np.argsort(connectivity)
    limit = min(max_candidates, len(order))
    return order[:limit].astype(int).tolist()


def build_transition_graph(score_matrix: np.ndarray, top_k: int = 8) -> List[np.ndarray]:
    """
    Build a sparse top-k neighbor graph from the dense score matrix.

    Args:
        score_matrix: Dense transition score matrix.
        top_k: Number of outgoing candidates to keep per frame.

    Returns:
        Per-node arrays of candidate next frames sorted by descending score.
    """
    n = score_matrix.shape[0]
    graph: List[np.ndarray] = []
    for i in range(n):
        row = score_matrix[i].copy()
        valid = np.where(row > INVALID_SCORE / 10)[0]
        valid = valid[valid != i]
        if valid.size == 0:
            graph.append(np.array([], dtype=np.int32))
            continue

        if valid.size > top_k:
            best = valid[np.argpartition(row[valid], -top_k)[-top_k:]]
        else:
            best = valid
        best = best[np.argsort(row[best])[::-1]]
        graph.append(best.astype(np.int32))
    return graph


def beam_search_sequence(
    score_matrix: np.ndarray,
    start_idx: int,
    graph: List[np.ndarray],
    beam_width: int = 6,
    penalty_weight: float = 35.0,
    lookahead_weight: float = 0.35,
):
    """
    Decode a frame ordering using beam search on the sparse transition graph.

    Args:
        score_matrix: Dense transition score matrix.
        start_idx: Seed frame.
        graph: Top-k adjacency list.
        beam_width: Number of partial hypotheses to keep per expansion.
        penalty_weight: Penalty for abrupt direction reversals.
        lookahead_weight: Weight for one-step future potential.

    Returns:
        Best decoded sequence.
    """
    n = score_matrix.shape[0]
    if n == 0:
        return []

    initial_used = np.zeros(n, dtype=bool)
    initial_used[start_idx] = True
    beams = [(0.0, [start_idx], initial_used)]

    for _ in range(1, n):
        candidates_for_round = []
        for beam_score, seq, used in beams:
            last = seq[-1]
            prev = seq[-2] if len(seq) > 1 else None

            neighbors = [int(j) for j in graph[last] if not used[j]]
            if not neighbors:
                fallback = np.where(~used)[0]
                fallback = fallback[fallback != last]
                neighbors = fallback.tolist()

            local_expansions = []
            for nxt in neighbors:
                transition_score = float(score_matrix[last, nxt])
                if prev is not None and (last - prev) * (nxt - last) < 0:
                    transition_score -= penalty_weight

                future_candidates = [int(j) for j in graph[nxt] if not used[j] and j != nxt]
                if future_candidates:
                    lookahead = float(np.max(score_matrix[nxt, future_candidates]))
                else:
                    remaining = np.where(~used)[0]
                    remaining = remaining[remaining != nxt]
                    lookahead = float(np.max(score_matrix[nxt, remaining])) if remaining.size else 0.0

                total = beam_score + transition_score + lookahead_weight * lookahead
                next_used = used.copy()
                next_used[nxt] = True
                local_expansions.append((total, seq + [nxt], next_used))

            if local_expansions:
                local_expansions.sort(key=lambda item: item[0], reverse=True)
                candidates_for_round.extend(local_expansions[:beam_width])

        if not candidates_for_round:
            break

        candidates_for_round.sort(key=lambda item: item[0], reverse=True)
        beams = candidates_for_round[:beam_width]

    best_score, best_sequence, _ = max(beams, key=lambda item: item[0])
    _ = best_score
    return best_sequence


def refine_sequence(sequence, score_matrix):
    """
    Run local sequence refinements in a stable order.
    """
    sequence = two_opt(sequence, score_matrix)
    sequence = smooth_temporal_coherence(sequence, score_matrix)
    sequence = remove_weak_links(sequence, score_matrix, drop_thresh=0.2)
    sequence = insert_missing_frames(sequence, score_matrix)
    sequence = two_opt(sequence, score_matrix)
    return sequence


def find_best_sequence(
    score_matrix: np.ndarray,
    penalty_weight: float = 35.0,
    lookahead_weight: float = 0.5,
    max_starts: int = 8,
    top_k: int = 8,
    beam_width: int = 6,
):
    """
    Search multiple plausible endpoints and keep the best sequence.
    """
    candidates = select_start_candidates(score_matrix, max_candidates=max_starts)
    if not candidates:
        return []

    graph = build_transition_graph(score_matrix, top_k=top_k)
    best_sequence = None
    best_score = -np.inf

    for start_idx in candidates:
        sequence = beam_search_sequence(
            score_matrix=score_matrix,
            start_idx=start_idx,
            graph=graph,
            beam_width=beam_width,
            penalty_weight=penalty_weight,
            lookahead_weight=lookahead_weight,
        )
        sequence = refine_sequence(sequence, score_matrix)
        if not sequence:
            continue

        direct_score = total_sequence_score(sequence, score_matrix)
        reverse_sequence = sequence[::-1]
        reverse_score = total_sequence_score(reverse_sequence, score_matrix)

        if reverse_score > direct_score:
            sequence = reverse_sequence
            candidate_score = reverse_score
        else:
            candidate_score = direct_score

        if candidate_score > best_score:
            best_sequence = sequence
            best_score = candidate_score

    return best_sequence if best_sequence is not None else []

def smooth_temporal_coherence(sequence, score_matrix, passes=2):
    """
    Improve ordering by swapping adjacent frames with higher match scores.

    Args:
        sequence: List of frame indices.
        score_matrix: Frame-to-frame score matrix.
        passes: Number of smoothing passes.

    Returns:
        Smoothed frame sequence.
    """
    seq = sequence.copy()
    for _ in range(passes):
        for i in range(1, len(seq) - 2):
            a, b, c = seq[i - 1], seq[i], seq[i + 1]
            original_score = score_matrix[a, b] + score_matrix[b, c]
            swapped_score = score_matrix[a, c] + score_matrix[c, b]
            monotonic_penalty = abs(b - a) if b < a and c > b else 0
            if swapped_score - original_score > monotonic_penalty:
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
    return seq

def remove_weak_links(sequence, score_matrix, drop_thresh=0.3):
    """
    Remove transitions with low similarity relative to average.

    Args:
        sequence: Frame ordering.
        score_matrix: Similarity matrix.
        drop_thresh: Threshold for dropping weak transitions.

    Returns:
        Pruned sequence.
    """
    if not sequence:
        return []

    cleaned = [sequence[0]]
    for i in range(1, len(sequence)):
        prev = cleaned[-1]
        curr = sequence[i]
        score = score_matrix[prev, curr]
        neighbor_scores = score_matrix[prev]
        valid_neighbors = neighbor_scores[neighbor_scores > INVALID_SCORE / 10]
        if valid_neighbors.size == 0:
            continue
        baseline = float(np.quantile(valid_neighbors, 0.6))
        threshold = drop_thresh * baseline
        if score >= threshold:
            cleaned.append(curr)
    return cleaned


def insert_missing_frames(sequence, score_matrix):
    """
    Reinsert omitted frames at the best scoring position.

    Cleanup should not permanently reduce coverage. If a frame is dropped by a
    conservative weak-link pass, insert it back where it hurts the sequence the
    least or improves it the most.
    """
    n = score_matrix.shape[0]
    if n == 0:
        return []
    if not sequence:
        return list(range(n))

    seq = sequence.copy()
    present = set(seq)
    missing = [idx for idx in range(n) if idx not in present]

    for node in missing:
        best_pos = len(seq)
        best_gain = -np.inf

        for pos in range(len(seq) + 1):
            if pos == 0:
                gain = float(score_matrix[node, seq[0]])
            elif pos == len(seq):
                gain = float(score_matrix[seq[-1], node])
            else:
                left = seq[pos - 1]
                right = seq[pos]
                gain = float(score_matrix[left, node] + score_matrix[node, right] - score_matrix[left, right])

            if gain > best_gain:
                best_gain = gain
                best_pos = pos

        seq.insert(best_pos, node)

    return seq

def total_sequence_score(sequence, score_matrix):
    """
    Calculate the total similarity score of a given frame sequence.

    Args:
        sequence: Frame ordering.
        score_matrix: Score matrix.

    Returns:
        Total sequence score.
    """
    return sum(score_matrix[sequence[i], sequence[i+1]] for i in range(len(sequence) - 1))

def two_opt(sequence, score_matrix):
    """
    Optimize the sequence using the 2-opt swapping heuristic.

    Args:
        sequence: Initial frame ordering.
        score_matrix: Pairwise score matrix.

    Returns:
        Improved sequence.
    """
    best_seq = sequence
    best_score = total_sequence_score(best_seq, score_matrix)
    improved = True
    while improved:
        improved = False
        # try all i<j pairs (skip endpoints)
        for i in range(1, len(best_seq)-2):
            for j in range(i+1, len(best_seq)-1):
                # reverse the segment between i and j
                candidate = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                cand_score = total_sequence_score(candidate, score_matrix)
                if cand_score > best_score:
                    best_seq, best_score = candidate, cand_score
                    improved = True
                    break
            if improved:
                break
    return best_seq

def greedy_with_lookahead(start_idx, score_matrix, penalty_weight=50.0, lookahead_weight=0.5):
    """
    Build frame sequence using greedy selection with lookahead.

    Args:
        start_idx: Index to start from.
        score_matrix: Frame similarity matrix.
        penalty_weight: Penalty for reversals.
        lookahead_weight: Weight for lookahead score.

    Returns:
        Frame sequence.
    """
    n = score_matrix.shape[0]
    used = np.zeros(n, dtype=bool)
    seq = [start_idx]
    used[start_idx] = True

    while len(seq) < n:
        last = seq[-1]
        prev = seq[-2] if len(seq) > 1 else None

        best_score = -np.inf
        best_next = None

        # Precompute available candidates
        candidates = np.where(~used)[0]
        for i in candidates:
            # immediate score
            score_now = score_matrix[last, i]

            # direction penalty
            if prev is not None:
                if (last - prev) * (i - last) < 0:
                    score_now -= penalty_weight

            # lookahead: best possible follow-on from i
            future_scores = score_matrix[i, candidates[candidates != i]]
            if future_scores.size:
                lookahead = future_scores.max()
            else:
                lookahead = 0.0

            total_score = score_now + lookahead_weight * lookahead
            if total_score > best_score:
                best_score = total_score
                best_next = i

        if best_next is None:
            break

        seq.append(best_next)
        used[best_next] = True

    return seq

def reconstruct_video(frames: List[np.ndarray], indices_order: List[int], output_path: str, fps: float = 30.0) -> None:
    """
    Write ordered frames to a video file with given FPS.

    Args:
        frames: video frames.
        indices_order: Desired order of frame indices.
        output_path: Output video path.
        fps: Output video frame rate.
    """
    if not frames:
        raise ValueError("No frames provided for reconstruction.")
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for idx in indices_order:
        writer.write(frames[idx])
    writer.release()
