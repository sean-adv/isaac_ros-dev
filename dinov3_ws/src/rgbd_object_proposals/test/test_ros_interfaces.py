"""Actual generated ROS messages, skipped only when ROS is unavailable."""
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

pytest.importorskip('rclpy')
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray
from rgbd_object_proposals.geometry import compute_object_obb
from rgbd_object_proposals.node import create_detection_msg
from dinov3_cases.proposals import draw_recognition, recognize_proposals


def test_message_and_crop_recognition():
    header = Header(frame_id='camera_color_optical_frame')
    header.stamp.sec = 42
    candidate = compute_object_obb(np.array([[0., 0., 1.], [.1, .1, 1.1], [.1, 0., 1.]]))
    candidate.bbox, candidate.candidate_id = (2, 4, 12, 14), 'candidate_0'
    detection = create_detection_msg(candidate, header)
    assert detection.bbox.center.position.x == 7.
    proposals = Detection2DArray(header=header, detections=[detection])

    class Encoder:
        def heatmap(self, image, bank):
            assert image.size == (10, 10)
            return np.full((10, 10), .8, dtype=np.float32)

    scores, output = recognize_proposals(Encoder(), Image.new('RGB', (20, 20)), None, proposals, 'medicine', .7)
    assert len(output.detections) == 1
    assert output.header == header
    assert output.detections[0].id == 'candidate_0'
    assert output.detections[0].results[0].hypothesis.class_id == 'medicine'
    assert output.detections[0].results[0].hypothesis.score == pytest.approx(.8)
    assert scores[0, 0] == -1

    lower_match = deepcopy(detection)
    lower_match.id = 'candidate_lower'
    lower_match.bbox.center.position.x = 15.
    proposals.detections = [lower_match, detection]

    class RankedEncoder:
        def __init__(self):
            self.similarities = iter((.75, .9))

        def heatmap(self, image, bank):
            return np.full((10, 10), next(self.similarities), dtype=np.float32)

    _, highest = recognize_proposals(
        RankedEncoder(), Image.new('RGB', (20, 20)), None, proposals, 'medicine', .7)
    assert len(highest.detections) == 1
    assert highest.detections[0].id == 'candidate_0'
    assert highest.detections[0].results[0].hypothesis.score == pytest.approx(.9)
    debug_image = np.zeros((20, 20, 3), dtype=np.uint8)
    draw_recognition(debug_image, highest)
    assert tuple(debug_image[4, 2]) == (0, 0, 255)

    proposals.detections = [detection]
    _, rejected = recognize_proposals(Encoder(), Image.new('RGB', (20, 20)), None, proposals, 'medicine', .9)
    assert rejected.detections == []
    proposals.detections = []
    _, empty = recognize_proposals(Encoder(), Image.new('RGB', (20, 20)), None, proposals, 'medicine', .7)
    assert empty.detections == []
