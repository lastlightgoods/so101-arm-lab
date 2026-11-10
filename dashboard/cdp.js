// node cdp.js shot <out.png> | eval "<js expr>"   — drives the kiosk Chrome via its DevTools port 9333 (first page target)
const [,, cmd, arg] = process.argv; const fs = require('fs');
(async () => {
  const targets = await (await fetch('http://127.0.0.1:9333/json')).json(); const t = targets.find(x => x.type === 'page' && x.url.includes('8765')) || targets[0];
  const ws = new WebSocket(t.webSocketDebuggerUrl); let id = 0; const pending = {};
  const send = (method, params = {}) => new Promise(r => { const i = ++id; pending[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
  ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && pending[m.id]) { pending[m.id](m.result || m.error); delete pending[m.id]; } };
  await new Promise(r => ws.onopen = r);
  if (cmd === 'shot') { const r = await send('Page.captureScreenshot', { format: 'png' }); fs.writeFileSync(arg, Buffer.from(r.data, 'base64')); console.log('saved', arg); }
  else { const r = await send('Runtime.evaluate', { expression: arg, returnByValue: true, awaitPromise: true }); console.log(JSON.stringify(r.result && r.result.value !== undefined ? r.result.value : r)); }
  ws.close();
})().catch(e => { console.error('cdp error', e.message); process.exit(1); });
