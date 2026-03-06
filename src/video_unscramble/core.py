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

    if not mini:
        km = KMeans(n_clusters=n_clusters, n_init=10, max_iter=max_iter, random_state=42)
        labels = km.fit_predict(features)
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,      
            max_iter=100,   
            tol=1e-4,
            batch_size=256,
            random_state=42,
            verbose=1,
        )
        for _ in tqdm(range(max_iter), desc="Clustering"):
            kmeans.partial_fit(features)
        labels = kmeans.predict(features)

    return labels

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

    n = len(frames)
    match_mat = np.zeros((n, n), dtype=np.float32)
    motion_mat = np.full((n, n), np.nan, dtype=np.float32)

    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) - 1)

    def worker(i: int, j: int):
        cnt, med = match_resnet_spatial_pair(feats[i], feats[j], ratio_thresh)
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

    if local_counts.max() > 0:
        local_norm = local_counts / local_counts.max()
    else:
        local_norm = local_counts.copy()

    # Weighted fusion
    return alpha * global_scores + (1.0 - alpha) * local_norm

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
    cleaned = [sequence[0]]
    for i in range(1, len(sequence)):
        prev = cleaned[-1]
        curr = sequence[i]
        score = score_matrix[prev, curr]
        avg = np.nanmean(score_matrix[prev])
        if score >= drop_thresh * avg:
            cleaned.append(curr)
    return cleaned

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

