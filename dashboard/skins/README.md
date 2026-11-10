# Dashboard skins

A skin is a palette, a font pair, panel treatments and an attract piece for the wall dashboard (`../index.html`).
The page is built on CSS tokens on `:root` (`--bg --panel --line --rim --ink --dim --acc --acc2 --warn --bad
--font --mono --disp --glow`); the gauges, cores, sparklines and status colours all read those tokens at draw
time (`css('--acc')` etc.), so recolouring is free. The one bespoke element is the attract canvas
(`#gallery > canvas#tempest` plus the HUD `#thud`), which each skin replaces with its own module.

## Files

| file | role |
|---|---|
| `skins.js` | the switcher: picks the skin (`?skin=name`, then localStorage `so101skin`, then `tempest`), loads `skins/<name>.css`, runs the skin module, cycles skins on the `k` key |
| `<name>.css` | tokens + treatments, scoped under `html[data-skin="<name>"]`. Loads its own fonts with `@import` (Google Fonts). |
| `<name>.js` | optional: `SKINS.register("<name>", {attract: {start(canvas, api), stop()}, labels: {"#id": "TEXT"}, apply(){}, remove(){}})` |

Everything is keyed on `html[data-skin]` so two skins never fight. The default skin `tempest` is the page as it
is today (its CSS lives in index.html; `tempest.css` is empty on purpose).

## Attract module contract

`start(canvas, api)` is called once with the `#tempest` canvas (already sized to the panel; redraw on `resize`
events) and `api = {state(), arms(), hist(), record(), lock()}` returning the latest polled JSON, or `null`.
Draw with `requestAnimationFrame`; keep CPU low (the wall kiosk runs all day): target <= 30 fps, no per-frame
allocations in the hot loop, stop the loop in `stop()`. The HUD overlay `#thud` is hidden for every skin except
tempest; a skin that wants a HUD draws it on the canvas or adds its own element inside `#gallery` and removes it
in `stop()`.

## Label swaps

`labels` maps element ids (or selectors) to replacement text for the skin's language ("MISSION STATUS" ->
"FLIGHT DIRECTOR"). The switcher stores the original text once and restores it when the skin changes. Only
static headings and button captions are swappable; live text (`#rectext`, gauge values) is never touched.

## The three skins

| skin | brief |
|---|---|
| `missioncontrol` | 1969 flight controller console. Cream phosphor on charcoal, IBM-style mono, seven-segment numerals, bezels. Needle meters drawn over the gauge arcs. Events log as a teletype. Record strip as a GO / NO-GO board. Attract: the last episode's joint trajectories plotted on graph paper. |
| `nostromo` | Green CRT text terminal with scanlines, typewriter reveals, Semiotic-Standard style icons on panel headers. Wireframe-only 3D arm. Lock screen asks for clearance. Attract: motion tracker, sweeping arc with arms and cameras as blips, ping cadence rises when recording. |
| `dystopian` | Lastlight Goods DYSTOPIAN sub-brand. **Silver only, never gold**: silver `#aeb4bb`, dusk `#3d3b49`, ink `#1a191f`, paper `#fdfbf6` for the rare light block. JetBrains Mono everywhere. Flat declarative copy ("ARM BAY. RECORDING. 12 OF 20."), stencil-weight headings, hazard striping on the record strip, facility signage ids (SECTOR 01 / ARM BAY / DATA VAULT). Lock screen as a badge reader with the all-silver ridge mark. Attract: facility status cards cycling, and the silver ridge polyline `(0,.16)(.17,.52)(.30,.34)(.46,.66)(.58,.44)(.72,1)(.84,.62)(1,.30)` drawn as a horizon with the live arm pose as a silhouette. |

## Orientation

Each skin must work in portrait (1440x2560, the wall) and landscape; the grid switch lives in index.html's
`@media (orientation: landscape)` block, skins only restyle.
