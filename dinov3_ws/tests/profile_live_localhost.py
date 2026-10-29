"""Sample an already running pipeline and receive its final WebRTC video.

No model loading, service restart, or grounding setting changes.
Status polling is best effort; short-lived results may be missed.
WebRTC times measure this receiver, not browser display latency.
"""
import argparse
import asyncio, json, time, urllib.request, pathlib
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8090')
parser.add_argument('--seconds', type=float, default=60)
parser.add_argument('--output', type=pathlib.Path, required=True)
args = parser.parse_args()
if args.seconds <= 0:
    parser.error('--seconds must be positive')
OUT=args.output
OUT.parent.mkdir(parents=True, exist_ok=True)
BASE=args.url.rstrip('/')
def req(path, data=None, method=None):
    r=urllib.request.Request(BASE+path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type':'application/json'},method=method)
    with urllib.request.urlopen(r, timeout=5) as f:
        b=f.read()
        return json.loads(b) if b else None
async def main():
    pc=RTCPeerConnection(RTCConfiguration(iceServers=[])); samples=[]; frames=[]; events={}; peer=None
    ready=asyncio.get_running_loop().create_future()
    @pc.on('track')
    def on_track(track): ready.set_result(track)
    pc.addTransceiver('video',direction='recvonly')
    try:
        await pc.setLocalDescription(await pc.createOffer())
        answer=await asyncio.to_thread(req,'/api/video/offer',{'sdp':pc.localDescription.sdp,'type':'offer'})
        peer=answer['peer_id']
        await pc.setRemoteDescription(RTCSessionDescription(answer['sdp'],answer['type']))
        track=await asyncio.wait_for(ready,10)
        async def receive():
            while True:
                f=await track.recv()
                frames.append({'at':time.monotonic(),'pts':f.pts,'time_base':str(f.time_base),'width':f.width,'height':f.height})
        task=asyncio.create_task(receive()); start=time.monotonic(); beat=start
        while time.monotonic()-start<args.seconds:
            before=time.monotonic(); s=await asyncio.to_thread(req,'/api/status'); now=time.monotonic()
            samples.append({'at':now,'http_ms':(now-before)*1000,'status':s})
            for e in s.get('sam3_inference_events',{}).get('events',[]): events[e['sequence']]=e
            if now-beat>10:
                await asyncio.to_thread(req,f'/api/video/peers/{peer}/heartbeat',{},'POST'); beat=now
            await asyncio.sleep(max(0,.02-(time.monotonic()-before)))
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        OUT.write_text(json.dumps({'start':start,'end':time.monotonic(),'samples':samples,'video_frames':frames,'sam3_events':list(events.values())},indent=2))
        print(json.dumps({'output':str(OUT),'polls':len(samples),'video_frames':len(frames)}))
    finally:
        await pc.close()
        if peer: await asyncio.to_thread(req,f'/api/video/peers/{peer}',None,'DELETE')
if __name__ == '__main__':
    asyncio.run(main())
