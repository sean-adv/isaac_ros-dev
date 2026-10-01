"""Generate a deterministic RGB-D scene by ray casting two rotated medicine boxes."""
from pathlib import Path
import argparse
import cv2
import numpy as np

from rgbd_object_proposals.geometry import Config, generate_proposals
from rgbd_object_proposals.visualization import draw_debug_image


def make_scene():
    h, w = 360, 640
    intrinsics = (600.0, 600.0, w / 2, h / 2)
    v, u = np.mgrid[:h, :w]
    rays = np.stack(((u - w / 2) / 600, (v - h / 2) / 600, np.ones((h, w))), axis=-1)
    depth = np.full((h, w), 1.1)
    rgb = np.full((h, w, 3), (55, 48, 42), dtype=np.uint8)
    masks = []
    ax, ay = .45, .50
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rotation = ry @ rx
    for center, size, color in [((-.15, 0, .88), (.13, .075, .055), (85, 155, 235)),
                                ((.15, .02, .85), (.11, .085, .06), (205, 160, 80))]:
        origin = -np.asarray(center) @ rotation
        direction = rays @ rotation
        half = np.array(size) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            t0, t1 = (-half - origin) / direction, (half - origin) / direction
        near, far = np.minimum(t0, t1).max(axis=2), np.maximum(t0, t1).min(axis=2)
        mask = (near > 0) & (far >= near) & (near < depth)
        depth[mask] = near[mask]
        rgb[mask] = color
        masks.append(mask)
    return rgb, depth.astype(np.float32), intrinsics, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/tmp/rgbd_proposals_demo.png')
    args = parser.parse_args()
    rgb, depth, intrinsics, _ = make_scene()
    cfg = Config(background_depth=1.1)
    candidates, _, _ = generate_proposals(depth, '32FC1', intrinsics, cfg)
    output = draw_debug_image(rgb, candidates)
    cv2.putText(output, 'SYNTHETIC / background subtraction + CC', (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                .6, (240, 240, 240), 1, cv2.LINE_AA)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(args.output, output):
        raise RuntimeError('Failed to save visualization')
    print(f'Candidates: {len(candidates)}; image: {args.output}')
    for candidate in candidates:
        print(candidate.candidate_id, candidate.bbox, candidate.dimensions)


if __name__ == '__main__':
    main()
