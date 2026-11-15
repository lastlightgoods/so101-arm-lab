// SO-101 Panel: the wall dashboard (http://127.0.0.1:8765) in its own native window, full screen on the vertical
// monitor, so it never shares a browser with the agents' Chrome (REQ 007). One WKWebView, no browser chrome.
// Built by dashboard/mac/build.sh into ~/Applications/SO-101 Panel.app; launched by `so101-dash app`.
//   SO101_PANEL_URL       page to load (default http://127.0.0.1:8765/)
//   SO101_PANEL_WINDOWED  1 = stay a normal window instead of going full screen
//   SO101_REVIEW_URL      TC Review page for Cmd-2 (default http://127.0.0.1:8790/, the private lab-board review server)
// Agent hooks (replacing cdp.js for this window): `kill -USR1 <app pid>` writes a PNG of the page to
// /tmp/so101-panel.png; the page's WebContent process id is in /tmp/so101-panel.webpid (for memory sampling).
// Safari's Develop menu can attach Web Inspector to it (isInspectable).
import Cocoa
import WebKit

let env = ProcessInfo.processInfo.environment
let pageURL = URL(string: env["SO101_PANEL_URL"] ?? "http://127.0.0.1:8765/")!
let reviewURL = URL(string: env["SO101_REVIEW_URL"] ?? "http://127.0.0.1:8790/")!   // TC Review (REQ 012): Cmd-2; Cmd-1 back to the arm panel
let shotPath = "/tmp/so101-panel.png"
let webPidPath = "/tmp/so101-panel.webpid"

final class App: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate, WKUIDelegate {
    var win: NSWindow!
    var web: WKWebView!
    var usr1: DispatchSourceSignal?
    var activity: NSObjectProtocol?
    var retryPending = false
    var current = pageURL

    // The vertical (portrait) screen first, else any screen without the menu bar, else the only one.
    func targetScreen() -> NSScreen {
        let s = NSScreen.screens
        return s.first(where: { $0.frame.height > $0.frame.width }) ?? s.dropFirst().first ?? s[0]
    }

    func applicationDidFinishLaunching(_ n: Notification) {
        buildMenu()
        let cfg = WKWebViewConfiguration()
        cfg.websiteDataStore = .default()                  // keeps the page's localStorage (skin, selections) between launches
        cfg.mediaTypesRequiringUserActionForPlayback = []
        web = WKWebView(frame: .zero, configuration: cfg)
        web.navigationDelegate = self
        web.uiDelegate = self
        if #available(macOS 13.3, *) { web.isInspectable = true }

        let f = targetScreen().visibleFrame
        win = NSWindow(contentRect: f, styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                       backing: .buffered, defer: false)
        win.title = "SO-101 Panel"
        win.titleVisibility = .hidden
        win.titlebarAppearsTransparent = true
        win.backgroundColor = NSColor(red: 0.02, green: 0.024, blue: 0.027, alpha: 1)   // the page's #050607
        win.collectionBehavior = [.fullScreenPrimary]
        win.isReleasedWhenClosed = false
        win.delegate = self
        win.contentView = web
        win.setFrame(f, display: true)
        win.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        if env["SO101_PANEL_WINDOWED"] != "1" { DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { self.win.toggleFullScreen(nil) } }

        // A live wall panel: no App Nap timer throttling; the display may still sleep as usual.
        activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiatedAllowingIdleSystemSleep], reason: "SO-101 live panel")

        signal(SIGUSR1, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: .main)
        src.setEventHandler { [weak self] in self?.snapshot() }
        src.resume()
        usr1 = src

        load()
    }

    func load() { web.load(URLRequest(url: current, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 10)) }
    @objc func showArmPanel(_ s: Any?) { current = pageURL; load() }
    @objc func showReview(_ s: Any?) { current = reviewURL; load() }

    // server.py not up yet (or restarting): keep retrying every 3 s instead of showing a dead page.
    func retrySoon() {
        if retryPending { return }
        retryPending = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) { self.retryPending = false; self.load() }
    }

    func webView(_ w: WKWebView, didFailProvisionalNavigation n: WKNavigation!, withError e: Error) { retrySoon() }
    func webView(_ w: WKWebView, didFail n: WKNavigation!, withError e: Error) { retrySoon() }

    // The page's process died (crash, or macOS killed it for memory): reload it rather than leave a blank screen.
    func webViewWebContentProcessDidTerminate(_ w: WKWebView) {
        NSLog("SO-101 Panel: web content process terminated, reloading")
        load()
    }

    func webView(_ w: WKWebView, didFinish n: WKNavigation!) {
        let sel = NSSelectorFromString("_webProcessIdentifier")   // private, read only for memory sampling; skipped if absent
        if w.responds(to: sel), let pid = w.value(forKey: "_webProcessIdentifier") as? Int32 {
            try? "\(pid)\n".write(toFile: webPidPath, atomically: true, encoding: .utf8)
        }
    }

    // Links that try to open a new window go to the default browser; the panel stays put.
    func webView(_ w: WKWebView, createWebViewWith c: WKWebViewConfiguration, for a: WKNavigationAction,
                 windowFeatures f: WKWindowFeatures) -> WKWebView? {
        if let u = a.request.url { NSWorkspace.shared.open(u) }
        return nil
    }

    func webView(_ w: WKWebView, runJavaScriptAlertPanelWithMessage m: String, initiatedByFrame f: WKFrameInfo,
                 completionHandler done: @escaping () -> Void) {
        let a = NSAlert(); a.messageText = m; a.addButton(withTitle: "OK")
        a.beginSheetModal(for: win) { _ in done() }
    }

    func webView(_ w: WKWebView, runJavaScriptConfirmPanelWithMessage m: String, initiatedByFrame f: WKFrameInfo,
                 completionHandler done: @escaping (Bool) -> Void) {
        let a = NSAlert(); a.messageText = m; a.addButton(withTitle: "OK"); a.addButton(withTitle: "Cancel")
        a.beginSheetModal(for: win) { r in done(r == .alertFirstButtonReturn) }
    }

    func snapshot() {
        web.takeSnapshot(with: nil) { img, err in
            guard let img = img, let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
                  let png = rep.representation(using: .png, properties: [:]) else {
                NSLog("SO-101 Panel: snapshot failed: \(String(describing: err))"); return
            }
            do { try png.write(to: URL(fileURLWithPath: shotPath)); NSLog("SO-101 Panel: snapshot \(shotPath)") }
            catch { NSLog("SO-101 Panel: snapshot write failed: \(error)") }
        }
    }

    @objc func reloadPage(_ s: Any?) { load() }

    func applicationShouldTerminateAfterLastWindowClosed(_ a: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ n: Notification) { try? FileManager.default.removeItem(atPath: webPidPath) }

    func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem(); main.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Arm Panel", action: #selector(showArmPanel(_:)), keyEquivalent: "1").target = self
        appMenu.addItem(withTitle: "TC Review", action: #selector(showReview(_:)), keyEquivalent: "2").target = self
        appMenu.addItem(withTitle: "Reload Panel", action: #selector(reloadPage(_:)), keyEquivalent: "r").target = self
        appMenu.addItem(withTitle: "Toggle Full Screen", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
            .keyEquivalentModifierMask = [.command, .control]
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit SO-101 Panel", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        let editItem = NSMenuItem(); main.addItem(editItem)
        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = edit
        NSApp.mainMenu = main
    }
}

let app = NSApplication.shared
let delegate = App()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
