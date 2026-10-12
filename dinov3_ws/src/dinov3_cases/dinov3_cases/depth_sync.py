"""Match registered RGB-D frames and camera intrinsics by size, frame and time."""


def stamp_seconds(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def matched_inputs(header, width, height, depths, infos, tolerance):
    """Only registered, same-size, timestamp-matched depth may use RGB boxes."""
    if not header.frame_id or stamp_seconds(header) <= 0:
        return None
    candidates = [
        depth for depth in depths
        if depth.header.frame_id == header.frame_id
        and (depth.width, depth.height) == (width, height)
        and stamp_seconds(depth.header) > 0
        and abs(stamp_seconds(depth.header) - stamp_seconds(header)) <= tolerance
    ]
    if not candidates:
        return None
    depth = min(candidates, key=lambda d: abs(stamp_seconds(d.header) - stamp_seconds(header)))
    for info in reversed(infos):
        if (info.header.frame_id == header.frame_id
                and (info.width, info.height) == (width, height)
                and stamp_seconds(info.header) <= stamp_seconds(header) + tolerance):
            return depth, info
    return None
