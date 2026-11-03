"""ROS message adapters for recognition results and camera-relative 3D boxes."""

import numpy as np
from geometry_msgs.msg import Point, Quaternion
from isaac_ros_apriltag_interfaces.msg import AprilTagDetection, AprilTagDetectionArray
from dinov3_cases.geometry.depth_bbox import _rotation_quaternion


def recognition_message(items, header=None):
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    message = Detection2DArray()
    if header is not None:
        message.header = header
    for item in items:
        if item["label"] == "unknown":
            continue
        x0, y0, x1, y1 = item["bbox"]
        detection = Detection2D()
        detection.header = message.header
        detection.id = str(item.get("track_id", ""))
        detection.bbox.center.position.x = (x0 + x1) / 2.0
        detection.bbox.center.position.y = (y0 + y1) / 2.0
        detection.bbox.size_x = float(x1 - x0)
        detection.bbox.size_y = float(y1 - y0)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = item["label"]
        hypothesis.hypothesis.score = item["score"] if item["score"] is not None else 0.0
        detection.results = [hypothesis]
        message.detections.append(detection)
    return message



def bboxes3d_message(items, header):
    """Standard detections sharing the final visualization's metric cuboid.

    Positions are meters in header.frame_id; dimensions are along local axes.
    The quaternion maps box-local axes to header.frame_id, matching visualization.
    Attached boxes are fitted to 2D detections; they do not measure the unseen
    physical center or thickness. Unfitted callers may also supply raw bounds.
    Covariance remains at the message default; uncertainty is not estimated.
    """
    from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
    from geometry_msgs.msg import Point, Quaternion

    message = Detection3DArray(header=header)
    for item in items:
        if not item.get("bbox_visible", True) or "bbox3d" not in item:
            continue
        data = item["bbox3d"]
        source = item.get("bbox3d_source")
        measured_header = header
        if source is not None:
            measured_header = type(header)(
                frame_id=source["frame_id"], stamp=type(header.stamp)(**source["stamp"]))
        detection = Detection3D(header=measured_header)
        detection.id = str(item.get("track_id", ""))
        center = detection.bbox.center
        x, y, z = map(float, data["position"])
        center.position = Point(x=x, y=y, z=z)
        qx, qy, qz, qw = map(float, _rotation_quaternion(data.get("rotation", np.eye(3))))
        center.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z = map(float, data["size"])
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = item["label"]
        result.hypothesis.score = float(item["score"] or 0.0)
        result.pose.pose.position = Point(x=x, y=y, z=z)
        result.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        detection.results = [result]
        message.detections.append(detection)
    if message.detections:
        # The array uses the newest measurement time; each detection retains its
        # own source time when cached objects were measured on different frames.
        message.header = max((item.header for item in message.detections),
                             key=lambda h: (h.stamp.sec, h.stamp.nanosec))
    return message



def apriltag_message(items, header):
    """Encode pixel bounds and camera-relative box poses for valid objects.

    family is the recognized class or unknown; id is the track ID, or 0 without tracking.
    Pose XYZ retain camera optical-frame coordinates in meters;
    pose Z is depth along the camera optical axis.
    corners are top-left, top-right, bottom-right, bottom-left in image pixels.
    Orientation uses the same estimated local box axes as Detection3D, including
    its camera-aligned fallback and surface-direction ambiguities. Pose headers
    retain the depth measurement time; pixel bounds use the array header.
    Covariance is left at its default (not estimated).
    """
    message = AprilTagDetectionArray(header=header)
    for item in items:
        if "bbox3d" not in item:
            continue
        detection = AprilTagDetection()
        detection.family = item["label"]
        detection.id = int(item.get("track_id", 0))
        x0, y0, x1, y1 = map(float, item["bbox"])
        detection.center = Point(x=(x0+x1)/2, y=(y0+y1)/2)
        detection.corners = [Point(x=x, y=y) for x, y in
                             ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        source = item.get("bbox3d_source")
        detection.pose.header = (header if source is None else type(header)(
            frame_id=source["frame_id"], stamp=type(header.stamp)(**source["stamp"])))
        data = item["bbox3d"]
        x, y, z = map(float, data["position"])
        detection.pose.pose.pose.position = Point(x=x, y=y, z=z)
        qx, qy, qz, qw = map(float, _rotation_quaternion(
            data.get("rotation", ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))))
        detection.pose.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        message.detections.append(detection)
    return message
