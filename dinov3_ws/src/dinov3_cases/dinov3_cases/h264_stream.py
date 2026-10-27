"""One Annex-B encoder, bounded GOP-aware fan-out, and WebRTC packet forwarding."""

import asyncio
from collections import deque
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from fractions import Fraction
import threading
import time

import av
from aiortc import (MediaStreamTrack, RTCConfiguration, RTCPeerConnection,
                    RTCRtpSender, RTCSessionDescription)
from aiortc.mediastreams import MediaStreamError

VIDEO_TIME_BASE = Fraction(1, 90000)


@dataclass(frozen=True)
class AccessUnit:
    data: bytes
    pts: int
    keyframe: bool
    header: object
    generation: int


class H264Encoder:
    """Owned by the rendering thread; no B frames or delayed output."""

    def __init__(self, codec="auto", fps=30, bitrate=2_000_000):
        if codec not in ("auto", "h264_nvenc", "libx264"):
            raise ValueError("h264_encoder must be auto, h264_nvenc or libx264")
        if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 30:
            raise ValueError("video_fps must be an integer in 1..30")
        if isinstance(bitrate, bool) or not isinstance(bitrate, int) or bitrate <= 0:
            raise ValueError("h264_bitrate must be a positive integer")
        self.requested, self.fps, self.bitrate = codec, fps, bitrate
        self.context = None
        self.codec = None
        self.last_pts = -1

    def _open(self, codec):
        context = av.CodecContext.create(codec, "w")
        context.width, context.height = 960, 540
        context.pix_fmt = "yuv420p"
        context.time_base, context.framerate = VIDEO_TIME_BASE, Fraction(self.fps)
        context.bit_rate, context.gop_size = self.bitrate, self.fps
        context.max_b_frames = 0
        context.thread_count = 2
        if codec == "h264_nvenc":
            context.options = dict(preset="p1", tune="ull", profile="baseline", level="3.1",
                                   **{"forced-idr": "1", "delay": "0", "rc-lookahead": "0",
                                      "zerolatency": "1"})
        else:
            context.options = dict(preset="ultrafast", tune="zerolatency", profile="baseline",
                                   level="3.1", **{"x264-params":
                                   "repeat-headers=1:annexb=1:scenecut=0"})
        context.open()
        return context

    def encode(self, pixels, header, generation, force_keyframe=False):
        if pixels.shape != (540, 960, 3):
            raise ValueError("H.264 input must be a 960x540 BGR image")
        pts = max(self.last_pts + 1, round(time.monotonic() * 90000))
        frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
        frame.pts, frame.time_base = pts, VIDEO_TIME_BASE
        if force_keyframe:
            frame.pict_type = av.video.frame.PictureType.I
        if self.context is None:
            candidates = ["h264_nvenc", "libx264"] if self.requested == "auto" else [self.requested]
            for codec in candidates:
                try:
                    context = self._open(codec)
                    packets = context.encode(frame)
                    if len(packets) != 1:
                        raise RuntimeError(f"{codec} buffered the frame; low-latency output required")
                    self.context, self.codec = context, codec
                    break
                except Exception:
                    if codec == candidates[-1]:
                        raise
        else:
            packets = self.context.encode(frame)
        if len(packets) != 1:
            raise RuntimeError("Expected one complete H.264 access unit per frame")
        self.last_pts = pts
        packet = packets[0]
        return AccessUnit(bytes(packet), pts, packet.is_keyframe, header, generation)

    def reset(self):
        # No delayed frames: dropping the context cannot discard pending output.
        self.context = None
        self.codec = None


class PacketReader:
    def __init__(self, hub, capacity):
        self.hub, self.capacity = hub, capacity
        self.condition = threading.Condition()
        self.queue = deque()
        self.waiting_keyframe = True
        self.closed = False

    def put(self, packet):
        with self.condition:
            if self.closed:
                return
            if len(self.queue) >= self.capacity:
                self.queue.clear()
                self.waiting_keyframe = True
                self.hub.keyframe_requested.set()
            if self.waiting_keyframe and not packet.keyframe:
                return
            self.waiting_keyframe = False
            self.queue.append(packet)
            self.condition.notify()

    def get(self, timeout=1):
        with self.condition:
            self.condition.wait_for(lambda: self.queue or self.closed, timeout)
            return self.queue.popleft() if self.queue else None

    def reset(self):
        with self.condition:
            self.queue.clear()
            self.waiting_keyframe = True
        self.hub.keyframe_requested.set()

    def close(self):
        with self.condition:
            self.closed = True
            self.queue.clear()
            self.condition.notify_all()
        self.hub.remove(self)


class PacketHub:
    def __init__(self, wake=lambda: None, capacity=30):
        self.lock = threading.Lock()
        self.readers = set()
        self.capacity, self.wake = capacity, wake
        self.keyframe_requested = threading.Event()

    @property
    def has_readers(self):
        with self.lock:
            return bool(self.readers)

    def subscribe(self):
        reader = PacketReader(self, self.capacity)
        with self.lock:
            self.readers.add(reader)
        self.keyframe_requested.set()
        self.wake()
        return reader

    def remove(self, reader):
        with self.lock:
            self.readers.discard(reader)

    def publish(self, packet):
        with self.lock:
            readers = tuple(self.readers)
        for reader in readers:
            reader.put(packet)

    def reset(self):
        with self.lock:
            readers = tuple(self.readers)
        for reader in readers:
            reader.reset()

    def close(self):
        with self.lock:
            readers = tuple(self.readers)
        for reader in readers:
            reader.close()


class EncodedVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, hub):
        super().__init__()
        self.reader = hub.subscribe()

    async def recv(self):
        while self.readyState == "live":
            unit = await asyncio.to_thread(self.reader.get)
            if unit is not None:
                packet = av.Packet(unit.data)
                packet.pts = packet.dts = unit.pts
                packet.time_base = VIDEO_TIME_BASE
                return packet  # aiortc packetizes this; it does not encode it again.
        raise MediaStreamError

    def stop(self):
        self.reader.close()
        super().stop()


class WebRTCVideo:
    """A dedicated asyncio loop; Flask handles only SDP signaling."""

    def __init__(self, hub, max_peers=8):
        self.hub, self.max_peers = hub, max_peers
        self.peers = {}
        self.last_seen = {}
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="webrtc", daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.run_until_complete(self.loop.shutdown_default_executor())
        self.loop.close()

    async def _close_peer(self, peer):
        track = self.peers.pop(peer, None)
        self.last_seen.pop(peer, None)
        if track is not None:
            track.stop()
        await peer.close()

    async def _offer(self, sdp, description_type):
        if description_type != "offer" or len(sdp) > 65536:
            raise ValueError("Expected a WebRTC SDP offer")
        if len(self.peers) >= self.max_peers:
            raise ValueError("Too many video viewers")
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        track = EncodedVideoTrack(self.hub)
        self.peers[peer] = track
        started_at = time.monotonic()
        self.last_seen[peer] = started_at

        @peer.on("connectionstatechange")
        async def state_changed():
            if peer.connectionState in ("failed", "closed", "disconnected"):
                await self._close_peer(peer)
            elif peer.connectionState == "connected":
                self.last_seen[peer] = time.monotonic()

        async def expire_inactive():
            # A killed tab may never send DELETE or a WebRTC close notification.
            while peer in self.peers:
                await asyncio.sleep(2)
                if peer not in self.peers:
                    return
                origin = self.last_seen[peer] if peer.connectionState == "connected" else started_at
                age = time.monotonic() - origin
                limit = 15 if peer.connectionState == "connected" else 30
                if age > limit:
                    await self._close_peer(peer)

        try:
            # Preferences must be set BEFORE setRemoteDescription: aiortc selects
            # the negotiated codec while applying the offer, not in createAnswer.
            peer.addTrack(track)
            codecs = [c for c in RTCRtpSender.getCapabilities("video").codecs
                      if c.mimeType.lower() == "video/h264"]
            for transceiver in peer.getTransceivers():
                if transceiver.kind == "video":
                    transceiver.direction = "sendonly"
                    transceiver.setCodecPreferences(codecs)
            await peer.setRemoteDescription(RTCSessionDescription(sdp, description_type))
            if not any(t.kind == "video" and t.mid is not None for t in peer.getTransceivers()):
                raise ValueError("Offer must include a video receiver")
            await peer.setLocalDescription(await peer.createAnswer())
            self.loop.create_task(expire_inactive())
            return dict(sdp=peer.localDescription.sdp, type=peer.localDescription.type,
                        peer_id=track.id)
        except BaseException as error:
            await self._close_peer(peer)
            if isinstance(error, Exception):
                raise ValueError(f"Invalid or incompatible WebRTC offer: {error}") from error
            raise

    def offer(self, sdp, description_type):
        future = asyncio.run_coroutine_threadsafe(self._offer(sdp, description_type), self.loop)
        try:
            return future.result(timeout=15)
        except FutureTimeout:
            future.cancel()
            raise ValueError("WebRTC negotiation timed out") from None

    def heartbeat(self, peer_id):
        async def refresh():
            for peer, track in self.peers.items():
                if track.id == peer_id:
                    self.last_seen[peer] = time.monotonic()
                    return True
            return False
        return asyncio.run_coroutine_threadsafe(refresh(), self.loop).result(timeout=5)

    def disconnect(self, peer_id):
        async def close():
            for peer, track in tuple(self.peers.items()):
                if track.id == peer_id:
                    await self._close_peer(peer)
        asyncio.run_coroutine_threadsafe(close(), self.loop).result(timeout=5)

    def close(self):
        if not self.thread.is_alive():
            return

        async def close_all():
            for peer in tuple(self.peers):
                await self._close_peer(peer)
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(close_all(), self.loop).result(timeout=10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
