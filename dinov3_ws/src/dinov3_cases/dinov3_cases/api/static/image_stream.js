/* Reconnect the plain HTTP image stream after network errors or page restoration. */
(() => {
  const image = document.querySelector('[data-live-image]');
  if (!image) return;
  const url = image.getAttribute('src');
  let timer, stopped = false;
  const connect = () => { image.src = url + '?t=' + Date.now(); };
  image.addEventListener('error', () => {
    clearTimeout(timer);
    if (!stopped) timer = setTimeout(connect, 2000);
  });
  window.addEventListener('pagehide', () => {
    stopped = true;
    clearTimeout(timer);
    image.removeAttribute('src');
  });
  window.addEventListener('pageshow', () => {
    if (stopped) { stopped = false; connect(); }
  });
})();
