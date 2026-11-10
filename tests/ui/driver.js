// node driver.js <cdp-port> <url> [css-selector-to-hide] [cdn]
// Opens <url> in a fresh tab of the headless Chrome listening on <cdp-port> (DevTools /json + WebSocket, same
// pattern as dashboard/cdp.js), blocks all external traffic (CDNs, fonts) so only the inline script runs (pass a
// 4th arg "cdn" to let the three.js/urdf-loader CDN modules through for the 3D-stage tests; fonts stay blocked), then
// reads one JSON command per stdin line and answers one JSON line each:
//   {"id":n,"cmd":"eval","expr":"<js>"}   -> {"id":n,"value":<json>} | {"id":n,"error":"<description>"}  (promises awaited)
//   {"id":n,"cmd":"key","key":"p"}        -> real keydown/keyup via Input.dispatchKeyEvent
//   {"id":n,"cmd":"errors"}               -> {"id":n,"exceptions":[...],"console":[...]}  (Runtime.exceptionThrown + console.error/warn)
//   {"id":n,"cmd":"clear"}                -> forget collected errors
//   {"id":n,"cmd":"shot","path":"x.png"} -> PNG screenshot of the viewport written to path
//   {"id":n,"cmd":"close"}                -> close the tab and exit
const [, , port, url, hide, net] = process.argv;
const readline = require('readline');
const base = `http://127.0.0.1:${port}`;
const out = o => process.stdout.write(JSON.stringify(o) + '\n');
let ws, id = 0; const pending = {}, waiters = [], exceptions = [], consoleMsgs = [];
const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pending[i] = { res, rej }; ws.send(JSON.stringify({ id: i, method, params })); });
const waitFor = method => new Promise(r => waiters.push(m => { if (m.method === method) { r(m.params); return true; } return false; }));
const desc = d => (d.exception && d.exception.description) || d.text || 'exception';
(async () => {
  const t = await (await fetch(`${base}/json/new?about:blank`, { method: 'PUT' })).json();
  ws = new WebSocket(t.webSocketDebuggerUrl);
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pending[m.id]) { const p = pending[m.id]; delete pending[m.id]; m.error ? p.rej(new Error(m.error.message)) : p.res(m.result); return; }
    if (m.method === 'Runtime.exceptionThrown') exceptions.push(desc(m.params.exceptionDetails));
    else if (m.method === 'Runtime.consoleAPICalled' && (m.params.type === 'error' || m.params.type === 'warning'))
      consoleMsgs.push(m.params.type + ': ' + m.params.args.map(a => a.value !== undefined ? String(a.value) : (a.description || a.type)).join(' '));
    for (let i = waiters.length - 1; i >= 0; i--) if (waiters[i](m)) waiters.splice(i, 1);
  };
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = () => j(new Error('websocket failed')); });
  await send('Runtime.enable'); await send('Page.enable'); await send('Network.enable');
  await send('Network.setBlockedURLs', { urls: net === 'cdn' ? ['https://fonts.*', 'http://fonts.*'] : ['https://*', 'http://fonts.*'] });   // no CDN/font traffic: deterministic, offline-safe; the three.js module never starts (unless 'cdn')
  if (hide) await send('Page.addScriptToEvaluateOnNewDocument', { source: `addEventListener('DOMContentLoaded',()=>{const el=document.querySelector(${JSON.stringify(hide)});if(el)el.style.display='none'})` });   // e.g. the Tempest canvas: software-rasterised shadow blur would keep the main thread busy and slow every fetch/eval ~50x
  const loaded = waitFor('Page.loadEventFired');
  await send('Page.navigate', { url });
  await loaded;
  out({ ready: true, targetId: t.id });
  const rl = readline.createInterface({ input: process.stdin });
  for await (const line of rl) {
    if (!line.trim()) continue;
    let c; try { c = JSON.parse(line); } catch (e) { out({ error: 'bad json: ' + line }); continue; }
    try {
      if (c.cmd === 'eval') {
        const r = await send('Runtime.evaluate', { expression: c.expr, returnByValue: true, awaitPromise: true });
        if (r.exceptionDetails) out({ id: c.id, error: desc(r.exceptionDetails) });
        else out({ id: c.id, value: r.result.value === undefined ? null : r.result.value });
      } else if (c.cmd === 'key') {
        const k = c.key, up = k.toUpperCase(), code = k.length === 1 ? 'Key' + up : k, vk = k.length === 1 ? up.charCodeAt(0) : 0;
        await send('Input.dispatchKeyEvent', { type: 'keyDown', key: k, code, windowsVirtualKeyCode: vk, text: k.length === 1 ? k : undefined });
        await send('Input.dispatchKeyEvent', { type: 'keyUp', key: k, code, windowsVirtualKeyCode: vk });
        out({ id: c.id, value: true });
      } else if (c.cmd === 'errors') out({ id: c.id, exceptions: exceptions.slice(), console: consoleMsgs.slice() });
      else if (c.cmd === 'clear') { exceptions.length = 0; consoleMsgs.length = 0; out({ id: c.id, value: true }); }
      else if (c.cmd === 'shot') { const r = await send('Page.captureScreenshot', { format: 'png' }); require('fs').writeFileSync(c.path, Buffer.from(r.data, 'base64')); out({ id: c.id, value: c.path }); }
      else if (c.cmd === 'close') { try { await fetch(`${base}/json/close/${t.id}`); } catch (e) {} ws.close(); out({ id: c.id, value: true }); process.exit(0); }
      else out({ id: c.id, error: 'unknown cmd ' + c.cmd });
    } catch (e) { out({ id: c.id, error: e.message }); }
  }
  process.exit(0);
})().catch(e => { out({ fatal: e.message }); process.exit(1); });
