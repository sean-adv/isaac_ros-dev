"""Cat2Real centered white-square preparation followed by OpenCV bilinear resize."""
import cv2
import numpy as np
from PIL import Image

MODEL_ID = 'zhanganyi88/Cat2Real-DINOv3-384'
REVISION = 'a763964f2041722ffcc5bc4c72e3cb241530727e'


def prepare_cat2real_image(image, size=384):
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    height, width = rgb.shape[:2]
    side = max(height, width)
    left, top = (side-width)//2, (side-height)//2
    square = np.full((side,side,3),255,dtype=np.uint8)
    square[top:top+height,left:left+width] = rgb
    prepared = Image.fromarray(cv2.resize(square,(size,size),interpolation=cv2.INTER_LINEAR))
    # Keep exact crop geometry for mapping patch heatmaps back onto the original ROI.
    scale = size/side
    x0,y0 = round(left*scale),round(top*scale)
    x1,y1 = round((left+width)*scale),round((top+height)*scale)
    prepared.info['content_box'] = (x0,y0,max(x0+1,min(size,x1)),max(y0+1,min(size,y1)))
    return prepared
