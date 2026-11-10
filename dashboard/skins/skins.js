// Skin switcher for the SO-101 wall dashboard. See README.md in this folder for the contract.
// Picks the skin from ?skin=, then localStorage, then "tempest"; loads skins/<name>.css and skins/<name>.js,
// applies label swaps, and hands the attract canvas to the skin. Key "k" cycles skins.
(() => {
  const NAMES = ["tempest", "missioncontrol", "nostromo", "dystopian"];
  const mods = {}, originals = new Map(); let current = null, running = null, cssLink = null;
  const api = {
    state: () => window.so101State || null, arms: () => window.armsNow || null,
    hist: () => window.so101Hist || null, record: () => window.recState || null, lock: () => !!window.lockedNow,
  };
  function pick() {
    const q = new URLSearchParams(location.search).get("skin");
    let s = q; if (!s) { try { s = localStorage.getItem("so101skin"); } catch (e) {} }
    return NAMES.includes(s) ? s : "tempest";
  }
  function loadScript(name) {
    return new Promise(res => {
      if (mods[name] || name === "tempest") return res();
      const el = document.createElement("script"); el.src = `skins/${name}.js`; el.onload = res; el.onerror = res; document.head.appendChild(el);
    });
  }
  function swapLabels(labels) {
    for (const [sel, text] of Object.entries(labels || {})) {
      const el = document.querySelector(sel); if (!el) continue;
      if (!originals.has(sel)) originals.set(sel, el.firstChild && el.firstChild.nodeType === 3 ? el.firstChild.textContent : el.textContent);
      if (el.firstChild && el.firstChild.nodeType === 3) el.firstChild.textContent = text; else el.textContent = text;
    }
  }
  function restoreLabels() {
    for (const [sel, text] of originals) { const el = document.querySelector(sel); if (!el) continue; if (el.firstChild && el.firstChild.nodeType === 3) el.firstChild.textContent = text; else el.textContent = text; }
  }
  async function apply(name) {
    if (name === current) return;
    if (running) { try { running.stop(); } catch (e) {} running = null; }
    if (current && mods[current] && mods[current].remove) { try { mods[current].remove(); } catch (e) {} }
    restoreLabels();
    document.documentElement.dataset.skin = name;
    if (cssLink) cssLink.remove(); cssLink = null;
    if (name !== "tempest") { cssLink = document.createElement("link"); cssLink.rel = "stylesheet"; cssLink.href = `skins/${name}.css`; document.head.appendChild(cssLink); }
    await loadScript(name);
    const canvas = document.querySelector("#tempest"), hud = document.querySelector("#thud");
    if (name === "tempest" || !mods[name]) {
      if (hud) hud.style.display = ""; if (window.startTempest) { try { window.startTempest(); } catch (e) {} }
    } else {
      if (window.stopTempest) { try { window.stopTempest(); } catch (e) {} }
      if (hud) hud.style.display = "none";
      const m = mods[name]; swapLabels(m.labels); if (m.apply) { try { m.apply(); } catch (e) {} }
      if (m.attract && canvas) { running = m.attract; try { running.start(canvas, api); } catch (e) { console.error("skin attract", e); running = null; } }
    }
    current = name; try { localStorage.setItem("so101skin", name); } catch (e) {}
    document.dispatchEvent(new CustomEvent("skinchange", { detail: { skin: name } }));
  }
  window.SKINS = {
    register(name, mod) { mods[name] = mod; }, apply, current: () => current, names: NAMES,
    next() { apply(NAMES[(NAMES.indexOf(current) + 1) % NAMES.length]); },
  };
  addEventListener("keydown", ev => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return; const t = ev.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (ev.key === "k" || ev.key === "K") { ev.preventDefault(); window.SKINS.next(); }
  });
  addEventListener("DOMContentLoaded", () => apply(pick()));
})();
