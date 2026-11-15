#!/bin/zsh
# build.sh [out-dir]  — builds "SO-101 Panel.app" (PanelApp.swift, one WKWebView) into ~/Applications by default.
# Needs only the Xcode command line tools (swiftc). Local, unsigned-for-distribution build (ad-hoc signed by the linker).
set -e
HERE=${0:A:h}; OUT=${1:-$HOME/Applications}; APP="$OUT/SO-101 Panel.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -swift-version 5 -o "$APP/Contents/MacOS/SO101Panel" "$HERE/PanelApp.swift"
sips -s format icns "$HERE/../icon.png" --out "$APP/Contents/Resources/SO101.icns" >/dev/null 2>&1 || true
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>com.lastlight.so101-panel</string>
  <key>CFBundleName</key><string>SO-101 Panel</string>
  <key>CFBundleDisplayName</key><string>SO-101 Panel</string>
  <key>CFBundleExecutable</key><string>SO101Panel</string>
  <key>CFBundleIconFile</key><string>SO101</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
PLIST
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "$APP"
