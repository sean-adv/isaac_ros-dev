"""Isaac ROS output must survive serialization with the RGB-D contract intact."""

from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest

from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from rclpy.serialization import serialize_message, deserialize_message
from std_msgs.msg import Header

from dinov3_cases.ros.messages import apriltag_message
from dinov3_cases.ros.messages import bboxes3d_message


def object_detection(**changes):
    item = dict(label="medicine", score=.9, bbox=[10, 20, 50, 80],
                bbox3d=dict(position=[.1, -.2, 1.3]))
    item.update(changes)
    return item


@pytest.mark.parametrize("label", ["medicine", "unknown"])
def test_serialized_pixel_bounds_and_metric_pose_preserve_headers(label):
    header = Header(frame_id="camera_color_optical_frame")
    header.stamp.sec, header.stamp.nanosec = 123, 456
    message = apriltag_message([object_detection(track_id=7, label=label, score=None if label == "unknown" else .9)], header)
    received = deserialize_message(serialize_message(message), AprilTagDetectionArray)
    assert received == message
    assert received.header == header
    detection, = received.detections
    assert detection.family == label and detection.id == 7
    assert (detection.center.x, detection.center.y, detection.center.z) == (30., 50., 0.)
    assert [(p.x, p.y, p.z) for p in detection.corners] == [
        (10., 20., 0.), (50., 20., 0.), (50., 80., 0.), (10., 80., 0.)]
    assert detection.pose.header == header
    pose = detection.pose.pose.pose
    assert (pose.position.x, pose.position.y) == (.1, -.2)
    assert pose.position.z == pytest.approx(1.3)
    assert (pose.orientation.x, pose.orientation.y, pose.orientation.z,
            pose.orientation.w) == (0., 0., 0., 1.)
    assert list(detection.pose.pose.covariance) == [0.] * 36



@pytest.mark.parametrize("position", [[0., 0., 1.3], [.3, -.4, 1.2]])
def test_topic_z_is_camera_depth_without_modifying_source(position):
    item = object_detection(bbox3d=dict(position=position.copy()))
    detection, = apriltag_message([item], Header()).detections
    output = detection.pose.pose.pose.position
    assert (output.x, output.y) == tuple(position[:2])
    assert output.z == pytest.approx(position[2])
    assert item["bbox3d"]["position"] == position


@pytest.mark.parametrize("rotation, expected", [
    ([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
     [0., 0., 2 ** -.5, 2 ** -.5]),
    ([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], [1., 0., 0., 0.]),
    ([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [0., 0., 0., 1.]),
])
def test_measured_rotation_survives_serialization_and_matches_detection3d(rotation, expected):
    item = object_detection(bbox3d=dict(position=[.1, -.2, 1.3],
                                       size=[.2, .3, .1], rotation=rotation))
    header = Header(frame_id="camera_color_optical_frame")
    message = apriltag_message([item], header)
    received = deserialize_message(serialize_message(message), AprilTagDetectionArray)
    orientation = received.detections[0].pose.pose.pose.orientation
    actual = [orientation.x, orientation.y, orientation.z, orientation.w]
    # q and -q encode the same rotation.
    assert min(sum((a - e) ** 2 for a, e in zip(actual, expected)),
               sum((a + e) ** 2 for a, e in zip(actual, expected))) < 1e-12
    assert sum(value ** 2 for value in actual) == pytest.approx(1.)
    assert orientation == bboxes3d_message([item], header).detections[0].bbox.center.orientation


def test_cached_pose_retains_measurement_header_and_current_pixel_bounds_header():
    header = Header(frame_id="current_camera")
    header.stamp.sec = 20
    item = object_detection(bbox3d_source=dict(
        frame_id="measurement_camera", stamp=dict(sec=19, nanosec=123)))
    message = apriltag_message([item], header)
    received = deserialize_message(serialize_message(message), AprilTagDetectionArray)
    assert received.header == header
    pose_header = received.detections[0].pose.header
    assert pose_header.frame_id == "measurement_camera"
    assert (pose_header.stamp.sec, pose_header.stamp.nanosec) == (19, 123)
    assert header.stamp.sec == 20


@pytest.mark.parametrize("label", ["medicine", "unknown"])
def test_missing_depth_and_empty_frames_do_not_invent_positions(label):
    no_depth = object_detection(label=label)
    del no_depth["bbox3d"]
    assert not apriltag_message([no_depth], Header()).detections
    assert not apriltag_message([], Header()).detections
    detection, = apriltag_message([object_detection()], Header()).detections
    assert detection.id == 0


def test_publication_worker_publishes_empty_array_to_clear_previous_objects():
    from dinov3_cases.ros.node import CaseNode

    stop = threading.Event()
    tag_pub = Mock()
    tag_pub.get_subscription_count.return_value = 1
    tag_pub.publish.side_effect = lambda _: stop.set()
    inactive = Mock()
    inactive.get_subscription_count.return_value = 0
    header = Header(frame_id="camera_color_optical_frame")
    logger = Mock()
    node = SimpleNamespace(
        stop=stop, publish_changed=threading.Event(), lock=threading.Lock(),
        pending_publication=([], header, 0), generation=0,
        tag_pub=tag_pub, recognition_pub=inactive, bbox3d_pub=inactive,
        get_logger=lambda: logger,
    )
    node.publish_changed.set()
    worker = threading.Thread(target=CaseNode.run_publications, args=(node,))
    worker.start()
    worker.join(3)
    try:
        assert not worker.is_alive()
        tag_pub.publish.assert_called_once()
        message = tag_pub.publish.call_args.args[0]
        assert isinstance(message, AprilTagDetectionArray)
        assert message.header == header and not message.detections
        logger.error.assert_not_called()
    finally:
        stop.set()
        node.publish_changed.set()
        worker.join(3)
