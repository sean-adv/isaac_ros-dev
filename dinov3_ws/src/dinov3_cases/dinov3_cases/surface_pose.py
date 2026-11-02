"""Deterministic local-PCA plane grouping and orthogonal cuboid orientation.

No random plane sampling is used. Voxel centroids bound normal-estimation cost;
normal modes are split by spatial connectivity before coplanar groups merge.
"""
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _plane(points, quantile):
    """Refine a plane after removing both signed residual tails once."""
    def fit(samples):
        center = samples.mean(axis=0)
        local = samples - center
        values, axes = np.linalg.eigh(local.T @ local / len(samples))
        return center, np.maximum(values, 0.), axes[:, 0]

    center, values, normal = fit(points)
    if quantile:
        residual = (points - center) @ normal
        lo, hi = np.quantile(residual, [quantile, 1 - quantile])
        kept = points[(residual >= lo - 1e-12) & (residual <= hi + 1e-12)]
        if len(kept) >= 12:
            center, values, normal = fit(kept)
    if normal @ center < 0:
        normal = -normal
    return dict(center=center, normal=normal, rms=float(np.sqrt(values[0])),
                area=float(12 * np.sqrt(values[1] * values[2])),
                count=len(points), samples=points, eigenvalues=values[::-1])


def extract_planes(points, quantile, *, max_samples=3500):
    """Return spatially connected, trimmed planes supported by local normals."""
    extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    voxel = max(.0015, extent / 100)
    _, inverse = np.unique(np.floor(points / voxel).astype(np.int64), axis=0,
                           return_inverse=True)
    counts = np.bincount(inverse)
    samples = np.column_stack([np.bincount(inverse, weights=points[:, axis]) / counts
                               for axis in range(3)])
    if len(samples) > max_samples:
        samples = samples[np.linspace(0, len(samples) - 1, max_samples, dtype=int)]
    if len(samples) < 24:
        return []
    tree = cKDTree(samples)
    distances, neighbors = tree.query(samples, k=min(24, len(samples)))
    local = samples[neighbors]
    local -= local.mean(axis=1, keepdims=True)
    values, axes = np.linalg.eigh(np.einsum('nki,nkj->nij', local, local) / local.shape[1])
    normals = axes[:, :, 0]
    normals[(normals * samples).sum(axis=1) < 0] *= -1
    # Ignore edge neighborhoods, depth discontinuities and line-like samples.
    usable = ((values[:, 1] > 1e-10) & (values[:, 0] < .12 * values[:, 1])
              & (distances[:, -1] < max(.025, voxel * 12)))
    pending = np.flatnonzero(usable)
    minimum = max(18, int(np.ceil(len(samples) * .025)))
    link_radius = max(voxel * 3, min(.025, float(np.median(distances[:, 5])) * 2.5))
    planes = []
    for _ in range(12):
        if len(pending) < minimum:
            break
        # A fixed angular histogram identifies modes without chaining normals
        # through a rounded/noisy ridge from one face into the next.
        bins = np.rint(normals[pending] / .18).astype(int)
        _, labels, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
        peak = normals[pending[labels == np.argmax(counts)]].mean(axis=0)
        peak /= np.linalg.norm(peak)
        matched = normals[pending] @ peak > np.cos(np.deg2rad(22))
        indices, pending = pending[matched], pending[~matched]
        if len(indices) < minimum:
            continue
        group = samples[indices]
        pairs = cKDTree(group).query_pairs(link_radius, output_type='ndarray')
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(group), len(group)))
        _, components = connected_components(graph, directed=False)
        for label in np.unique(components):
            component = group[components == label]
            if len(component) < minimum:
                continue
            plane = _plane(component, quantile)
            if plane['rms'] > .004 or plane['area'] < voxel ** 2 * 20:
                continue
            # Merge fragmented coplanar faces, but not parallel offset faces
            # or disconnected objects on a distant background plane.
            for previous in planes:
                if (abs(previous['normal'] @ plane['normal']) > np.cos(np.deg2rad(10))
                        and abs((plane['center'] - previous['center']) @ previous['normal']) < .003
                        and cKDTree(previous['samples']).query(component)[0].min() < link_radius * 3):
                    previous.update(_plane(np.vstack((previous['samples'], component)), quantile))
                    break
            else:
                planes.append(plane)
    for plane in planes:
        plane['weight'] = plane['count'] / (1 + (plane['rms'] / .0015) ** 2)
    return sorted(planes, key=lambda p: p['weight'], reverse=True)[:8]


def _adjacent(a, b):
    gap = cKDTree(a['samples']).query(b['samples'])[0].min()
    scale = min(np.linalg.norm(np.ptp(a['samples'], axis=0)),
                np.linalg.norm(np.ptp(b['samples'], axis=0)))
    return gap < max(.012, scale * .2)


def choose_faces(planes):
    """Choose two/three neighboring faces consistent with a rectangular box."""
    best, best_score = [], 0.
    for count in (2, 3):
        for group in combinations(planes, count):
            dots = [abs(a['normal'] @ b['normal']) for a, b in combinations(group, 2)]
            if max(dots) > np.sin(np.deg2rad(20)):
                continue
            if not all(_adjacent(a, b) for a, b in combinations(group, 2)):
                continue
            score = sum(p['weight'] for p in group) * (1 - np.mean(np.square(dots)))
            if score > best_score:
                best, best_score = list(group), score
    return best


def support_mask(points, planes):
    """Keep points close to a selected face and its observed spatial support."""
    keep = np.zeros(len(points), dtype=bool)
    for plane in planes:
        residual = np.abs((points - plane['center']) @ plane['normal'])
        tolerance = max(.002, min(.006, plane['rms'] * 3))
        near = residual <= tolerance
        if near.any():
            distance = cKDTree(plane['samples']).query(points[near])[0]
            span = np.linalg.norm(np.ptp(plane['samples'], axis=0))
            keep[np.flatnonzero(near)[distance <= max(.01, span * .08)]] = True
    return keep


def orthogonal_rotation(faces, points):
    """Weighted closest orthogonal frame; canonicalize signs and axis order."""
    normals = np.column_stack([face['normal'] for face in faces])
    weights = np.array([face['weight'] for face in faces])
    weights /= weights.max()
    if len(faces) == 2:
        third = np.cross(normals[:, 0], normals[:, 1])
        third /= np.linalg.norm(third)
        normals = np.column_stack((normals, third))
        weights = np.r_[weights, .01]
    elif np.linalg.det(normals) < 0:
        normals[:, 2] *= -1
    u, _, vt = np.linalg.svd(normals * weights)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    spans = np.ptp(points @ rotation, axis=0)
    x_index = int(np.argmax(spans))
    others = [i for i in range(3) if i != x_index]
    z_index = max(others, key=lambda i: abs(rotation[:, i] @ points.mean(axis=0)))
    x, z = rotation[:, x_index].copy(), rotation[:, z_index].copy()
    if x[np.argmax(np.abs(x))] < 0:
        x *= -1
    if z @ points.mean(axis=0) < 0:
        z *= -1
    return np.column_stack((x, np.cross(z, x), z))


def face_diagnostics(planes):
    return [dict(normal=p['normal'].tolist(), center=p['center'].tolist(),
                 sample_count=p['count'], rms_m=p['rms'], area_m2=p['area']) for p in planes]
