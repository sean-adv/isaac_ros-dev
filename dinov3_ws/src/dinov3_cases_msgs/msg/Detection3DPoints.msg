std_msgs/Header header
string id
vision_msgs/ObjectHypothesis hypothesis
vision_msgs/BoundingBox3D bbox
vision_msgs/BoundingBox2D bbox_2d
# Original geometric corners, order: top-left, top-right, bottom-right, bottom-left.
# Right/bottom edges retain the exclusive bounds of the original 2D box.
dinov3_cases_msgs/Point2D[4] bbox_2d_corners
# Actual pixels found along each corner-to-center line, in the same order.
dinov3_cases_msgs/Point2D[4] sampled_2d_points
# Unprojected samples, meters in header.frame_id, in the same order.
geometry_msgs/Point[4] sampled_3d_points
# Minimum valid optical Z inside the 2D box (not minimum Euclidean distance).
dinov3_cases_msgs/Point2D nearest_2d_point
geometry_msgs/Point nearest_3d_point
# Final box vertices: min-Z face TL,TR,BR,BL, then max-Z face TL,TR,BR,BL.
geometry_msgs/Point[8] bbox_3d_corners
