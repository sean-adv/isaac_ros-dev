"""Measure camera delivery alone, with no inference or image copying."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seconds',type=float,default=10)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rclpy.init()
    node=rclpy.create_node('camera_input_profile')
    rows=[]
    def callback(msg):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        rows.append(dict(at=time.monotonic(),stamp=stamp,age_ms=(node.get_clock().now().nanoseconds*1e-9-stamp)*1000,width=msg.width,height=msg.height))
    node.create_subscription(Image,'/camera/color/image_raw',callback,qos_profile_sensor_data)
    end=time.monotonic()+args.seconds+2
    try:
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=0.05)
        if not rows: raise RuntimeError('No camera images received')
        rows=[r for r in rows if r['at']>=rows[0]['at']+1]
        summary=dict(samples=len(rows),callback_fps=(len(rows)-1)/(rows[-1]['at']-rows[0]['at']),stamp_fps=(len(rows)-1)/(rows[-1]['stamp']-rows[0]['stamp']),age_mean_ms=float(np.mean([r['age_ms'] for r in rows])),age_p95_ms=float(np.percentile([r['age_ms'] for r in rows],95)),image_size=[rows[0]['width'],rows[0]['height']])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n')
        print(json.dumps(summary,indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__': main()
