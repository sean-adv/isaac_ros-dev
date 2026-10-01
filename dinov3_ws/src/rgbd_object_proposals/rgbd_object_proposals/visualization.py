"""Debug rendering shared by the ROS node and the offline example."""
import cv2


def draw_debug_image(bgr, candidates):
    image = bgr.copy()
    for candidate in candidates:
        x0, y0, x1, y1 = candidate.bbox
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (60, 220, 80), 2)
        cv2.putText(image, candidate.candidate_id, (x0, max(16, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 80), 1, cv2.LINE_AA)
    return image
