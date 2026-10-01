"""Aligned RGB-D -> synchronized RGB + Detection2DArray proposals."""
from dataclasses import asdict

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from vision_msgs.msg import Detection2D, Detection2DArray

from .geometry import Config, generate_proposals
from .visualization import draw_debug_image


def create_detection_msg(candidate, header):
    detection = Detection2D()
    detection.header = header
    detection.id = candidate.candidate_id
    x0, y0, x1, y1 = candidate.bbox
    detection.bbox.center.position.x = (x0 + x1) / 2.0
    detection.bbox.center.position.y = (y0 + y1) / 2.0
    detection.bbox.size_x = float(x1 - x0)
    detection.bbox.size_y = float(y1 - y0)
    # No classification score: geometry is not a recognition probability.
    return detection


class RGBDObjectProposalNode(Node):
    def __init__(self):
        super().__init__('rgbd_object_proposal_node')
        defaults = dict(rgb_topic='/camera/color/image_raw', depth_topic='/camera/depth/image_raw',
                        camera_info_topic='/camera/color/camera_info',
                        sync_queue_size=10, sync_slop=0.05, camera_info_max_age=0.0,
                        publish_debug_image=True, publish_pointcloud=False)
        defaults.update(asdict(Config()))
        for name, value in defaults.items():
            self.declare_parameter(name, value, ParameterDescriptor(read_only=True))
        self.p = lambda name: self.get_parameter(name).value
        self.cfg = Config(**{name: self.p(name) for name in asdict(Config())})
        self.cfg.validate()
        if self.p('sync_queue_size') < 1 or not np.isfinite(self.p('sync_slop')) or self.p('sync_slop') < 0:
            raise ValueError('Invalid synchronization parameters')
        if not np.isfinite(self.p('camera_info_max_age')) or self.p('camera_info_max_age') < 0:
            raise ValueError('camera_info_max_age must be nonnegative')
        self.bridge = CvBridge()
        self.camera_info = None
        self.info_sub = self.create_subscription(CameraInfo, self.p('camera_info_topic'),
                                                 self.on_info, qos_profile_sensor_data)
        self.rgb_sub = message_filters.Subscriber(self, Image, self.p('rgb_topic'), qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.p('depth_topic'), qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], self.p('sync_queue_size'), self.p('sync_slop'))
        self.sync.registerCallback(self.on_images)
        self.detections_pub = self.create_publisher(Detection2DArray, '/rgbd_proposals/detections', 10)
        self.rgb_pub = self.create_publisher(Image, '/rgbd_proposals/rgb', 10)
        self.debug_pub = self.create_publisher(Image, '/rgbd_proposals/debug_image', 1)
        self.cloud_pub = self.create_publisher(PointCloud2, '/rgbd_proposals/points', 1)
        self.get_logger().info(
            f'Using configured background depth {self.cfg.background_depth:.3f} m; depth must align to RGB.')

    def on_info(self, msg):
        self.camera_info = msg

    def on_images(self, rgb, depth):
        try:
            info = self.camera_info
            if info is None:
                raise ValueError('Waiting for color CameraInfo')
            if (rgb.width, rgb.height) != (depth.width, depth.height) or (rgb.width, rgb.height) != (info.width, info.height):
                raise ValueError('Aligned depth, RGB and CameraInfo must have equal image dimensions')
            if not rgb.header.frame_id or rgb.header.frame_id != depth.header.frame_id or rgb.header.frame_id != info.header.frame_id:
                raise ValueError('RGB, aligned depth and CameraInfo must share the color optical frame_id')
            if info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset:
                raise ValueError('Use full-resolution CameraInfo without binning or ROI offsets')
            distortion = None
            if not np.all(np.isfinite(info.d)):
                raise ValueError('Camera distortion coefficients must be finite')
            if any(abs(value) > 1e-8 for value in info.d):
                if info.distortion_model not in ('plumb_bob', 'rational_polynomial') or len(info.d) not in (4, 5, 8, 12, 14):
                    raise ValueError('Unsupported camera distortion; use plumb_bob/rational_polynomial or rectified inputs')
                distortion = info.d
            stamp = lambda msg: msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if self.p('camera_info_max_age') > 0 and abs(stamp(rgb) - stamp(info)) > self.p('camera_info_max_age'):
                raise ValueError('CameraInfo is too old for this RGB image')
            array = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
            intrinsics = (info.k[0], info.k[4], info.k[2], info.k[5])
            candidates, points, _ = generate_proposals(
                array, depth.encoding, intrinsics, self.cfg, distortion)
            detections = Detection2DArray()
            detections.header = rgb.header
            detections.detections = [create_detection_msg(c, rgb.header) for c in candidates]
            for candidate in candidates:
                x0, y0, x1, y1 = candidate.bbox
                self.get_logger().info(
                    f'Detected {candidate.candidate_id} bbox: '
                    f'x=[{x0}, {x1}), y=[{y0}, {y1})')
            # Republish this exact RGB message; consumers use exact timestamp synchronization.
            self.rgb_pub.publish(rgb)
            self.detections_pub.publish(detections)
            if self.p('publish_debug_image'):
                bgr = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
                debug = self.bridge.cv2_to_imgmsg(draw_debug_image(bgr, candidates), encoding='bgr8')
                debug.header = rgb.header
                self.debug_pub.publish(debug)
            if self.p('publish_pointcloud'):
                self.cloud_pub.publish(create_cloud_xyz32(rgb.header, points.astype(np.float32)))
        except (ValueError, TypeError, RuntimeError, CvBridgeError, cv2.error, np.linalg.LinAlgError) as error:
            self.get_logger().warning(str(error), throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RGBDObjectProposalNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
