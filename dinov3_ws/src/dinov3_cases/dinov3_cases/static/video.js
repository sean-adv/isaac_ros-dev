/* The server forwards one shared H.264 stream; browsers only decode it. */
(() => {
  const video = document.querySelector('[data-live-video]');
  if (!video) return;
  const status = document.querySelector('[data-video-status]');
  let peer = null, peerId = null, timer = null, heartbeat = null, stopped = false, attempt = 0;
  const show = text => { if (status) status.textContent = text; };

  function closePeer() {
    clearInterval(heartbeat);
    heartbeat = null;
    if (peer) {
      peer.onconnectionstatechange = null;
      peer.close();
      peer = null;
    }
    video.srcObject = null;
    if (peerId) {
      fetch('/api/video/peers/' + encodeURIComponent(peerId), {
        method: 'DELETE', keepalive: true
      }).catch(() => {});
      peerId = null;
    }
  }

  function retry(message) {
    if (stopped) return;
    ++attempt;
    closePeer();
    clearTimeout(timer);
    show(message + ' · reconnecting…');
    timer = setTimeout(connect, 2000);
  }

  async function connect() {
    if (stopped) return;
    if (!window.RTCPeerConnection) {
      show('This browser does not support WebRTC video.');
      return;
    }
    const token = ++attempt;
    closePeer();
    show('Connecting H.264 video…');
    const pc = new RTCPeerConnection({iceServers: []});
    peer = pc;
    let timeout;
    try {
      pc.addTransceiver('video', {direction: 'recvonly'});
      pc.ontrack = event => {
        if (token !== attempt) return;
        video.srcObject = new MediaStream([event.track]);
        video.play().catch(() => show('Select the video to start playback.'));
      };
      pc.onconnectionstatechange = () => {
        if (token !== attempt) return;
        if (pc.connectionState === 'connected') show('Connected · waiting for video…');
        if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) {
          retry('Video connection interrupted');
        }
      };
      await pc.setLocalDescription(await pc.createOffer());
      if (pc.iceGatheringState !== 'complete') {
        await new Promise((resolve, reject) => {
          timeout = setTimeout(() => reject(new Error('ICE gathering timed out')), 10000);
          pc.addEventListener('icegatheringstatechange', () => {
            if (pc.iceGatheringState === 'complete') resolve();
          });
        });
        clearTimeout(timeout);
      }
      const response = await fetch('/api/video/offer', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(pc.localDescription), signal: AbortSignal.timeout(20000)
      });
      const answer = await response.json();
      if (!response.ok) throw new Error(answer.error || 'Video negotiation failed');
      if (token !== attempt || stopped) {
        fetch('/api/video/peers/' + encodeURIComponent(answer.peer_id), {method: 'DELETE'});
        return;
      }
      peerId = answer.peer_id;
      heartbeat = setInterval(async () => {
        const id = peerId;
        if (!id) return;
        try {
          const response = await fetch('/api/video/peers/' + encodeURIComponent(id) + '/heartbeat', {
            method: 'POST', signal: AbortSignal.timeout(5000)
          });
          if (response.status === 404 && id === peerId) retry('Video session expired');
        } catch (_) { /* Connection-state handling and the server lease cover outages. */ }
      }, 3000);
      await pc.setRemoteDescription({type: answer.type, sdp: answer.sdp});
      timer = setTimeout(() => {
        if (token === attempt && pc.connectionState !== 'connected') retry('Video connection timed out');
      }, 15000);
    } catch (error) {
      clearTimeout(timeout);
      if (token === attempt) retry(error.message);
    }
  }

  video.addEventListener('playing', () => show('Live · H.264 · 960×540'));
  video.addEventListener('click', () => video.play().catch(() => {}));
  window.addEventListener('pagehide', () => {
    stopped = true;
    ++attempt;
    clearTimeout(timer);
    closePeer();
  });
  window.addEventListener('pageshow', () => {
    if (stopped) { stopped = false; connect(); }
  });
  connect();
})();
