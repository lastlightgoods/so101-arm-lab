# Security

This repository drives a physical robot arm and reads cameras on a home network. Please report anything that could
move the arm, expose a camera, or leak a credential **privately**, through a
[security advisory](https://github.com/lastlightgoods/so101-arm-lab/security/advisories/new) rather than a public issue.

What is in scope: the wall dashboard (`dashboard/server.py`), the LAN gateway (`dashboard/phone.py`, pad and phone
modes), the Jetson scripts under `jetson/`, and anything that handles tokens or passphrases.

What you should expect: an acknowledgement within a few days and a fix or a documented mitigation before details are
published. Secret scanning and push protection are on for this repository; if a secret ever lands here anyway, it
will be rotated, not just deleted.
