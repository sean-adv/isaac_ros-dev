"""Publish recognized RGB-D objects using the Isaac ROS AprilTag contract."""

from geometry_msgs.msg import Point
from isaac_ros_apriltag_interfaces.msg import AprilTagDetection, AprilTagDetectionArray


def apriltag_message(items, header):
    """Encode pixel bounds and camera-relative XYZ for valid objects.

    family is the recognized class; id is the track ID, or 0 without tracking.
    Pose XYZ retain camera optical-frame coordinates in meters;
    pose Z is depth along the camera optical axis.
    corners are top-left, top-right, bottom-right, bottom-left in image pixels.
    The identity quaternion describes camera-aligned box axes, not a measured
    object rotation. Covariance is left at its default (not estimated).
    """
    message = AprilTagDetectionArray(header=header)
    for item in items:
        if item["label"] == "unknown" or "bbox3d" not in item:
            continue
        detection = AprilTagDetection()
        detection.family = item["label"]
        detection.id = int(item.get("track_id", 0))
        x0, y0, x1, y1 = map(float, item["bbox"])
        detection.center = Point(x=(x0+x1)/2, y=(y0+y1)/2)
        detection.corners = [Point(x=x, y=y) for x, y in
                             ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        detection.pose.header = header
        x, y, z = map(float, item["bbox3d"]["position"])
        detection.pose.pose.pose.position = Point(x=x, y=y, z=z)
        detection.pose.pose.pose.orientation.w = 1.0
        message.detections.append(detection)
    return message
