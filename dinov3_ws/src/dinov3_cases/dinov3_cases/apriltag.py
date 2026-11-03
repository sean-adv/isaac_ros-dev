"""Publish RGB-D objects using the Isaac ROS AprilTag contract."""

from geometry_msgs.msg import Point, Quaternion
from isaac_ros_apriltag_interfaces.msg import AprilTagDetection, AprilTagDetectionArray

from .depth_bbox import _rotation_quaternion


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
