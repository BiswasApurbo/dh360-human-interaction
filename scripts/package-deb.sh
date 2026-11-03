#!/usr/bin/env bash
# ==============================================================================
# KBCON ARM64 Debian Package (.deb) Builder
# Builds kbcon-device-agent_<version>_arm64.deb for Raspberry Pi 5
# ==============================================================================

set -euo pipefail

PACKAGE_NAME="kbcon-device-agent"
VERSION="1.0.0"
ARCH="arm64"
DIST_DIR="dist"
BUILD_ROOT="${DIST_DIR}/${PACKAGE_NAME}_${VERSION}_${ARCH}"

echo "Building Debian package: ${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"

rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}/DEBIAN"
mkdir -p "${BUILD_ROOT}/etc/kbcon/device"
mkdir -p "${BUILD_ROOT}/etc/modprobe.d"
mkdir -p "${BUILD_ROOT}/lib/systemd/system"
mkdir -p "${BUILD_ROOT}/usr/local/bin"
mkdir -p "${BUILD_ROOT}/opt/kbcon"

# 1. DEBIAN Control File
cat > "${BUILD_ROOT}/DEBIAN/control" << EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Depends: python3 (>= 3.11), python3-venv, python3-pip, v4l-utils, git, curl, libgl1, libglib2.0-0
Maintainer: KBCON Robotics Team <support@kbcon.ai>
Description: KBCON Humanoid Robot Hardware Provisioning & Device Agent
 Automated hardware provisioning, asymmetric cryptographic identity,
 Sonix camera driver fix, and autonomous cloud pairing for Raspberry Pi 5.
EOF

# 2. Camera Blacklist
cat > "${BUILD_ROOT}/etc/modprobe.d/blacklist-option.conf" << 'EOF'
# KBCON / DH360 Hardware Fix: Prevent GSM USB serial driver (option) from
# hijacking Sonix MWB-16 USB video camera (ID 1410:1410).
blacklist option
EOF
chmod 644 "${BUILD_ROOT}/etc/modprobe.d/blacklist-option.conf"

# 3. Systemd Service
cat > "${BUILD_ROOT}/lib/systemd/system/dh360-backend.service" << 'EOF'
[Unit]
Description=KBCON / DH360 Human Interaction Subsystem & Backend
After=network.target network-online.target sound.target

[Service]
Type=simple
User=eliva
Group=eliva
WorkingDirectory=/home/eliva/dh360-human-interaction
Environment="PATH=/home/eliva/dh360-human-interaction/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/eliva/dh360-human-interaction/.venv/bin/python -m dh360_hi.cli.main serve --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
chmod 644 "${BUILD_ROOT}/lib/systemd/system/dh360-backend.service"

# 4. Global kbcon-device wrapper
cat > "${BUILD_ROOT}/usr/local/bin/kbcon-device" << 'EOF'
#!/usr/bin/env bash
if [ -x /home/eliva/dh360-human-interaction/.venv/bin/kbcon-device ]; then
    exec /home/eliva/dh360-human-interaction/.venv/bin/kbcon-device "$@"
else
    exec python3 -m dh360_hi.cli.device "$@"
fi
EOF
chmod 755 "${BUILD_ROOT}/usr/local/bin/kbcon-device"

# 5. Post-installation script (postinst)
cat > "${BUILD_ROOT}/DEBIAN/postinst" << 'EOF'
#!/bin/sh
set -e

chmod 700 /etc/kbcon/device 2>/dev/null || true
if id "eliva" >/dev/null 2>&1; then
    chown -R eliva:eliva /etc/kbcon/device 2>/dev/null || true
fi

# Reload systemd
if [ -d /run/systemd/system ]; then
    systemctl daemon-reload || true
    systemctl enable dh360-backend.service || true
fi

echo "=============================================================="
echo " KBCON Device Agent installed successfully."
echo " Run 'kbcon-device pair' to link this Raspberry Pi to a robot."
echo "=============================================================="
EOF
chmod 755 "${BUILD_ROOT}/DEBIAN/postinst"

# 6. Pre-removal script (prerm)
cat > "${BUILD_ROOT}/DEBIAN/prerm" << 'EOF'
#!/bin/sh
set -e
if [ -d /run/systemd/system ]; then
    systemctl stop dh360-backend.service 2>/dev/null || true
    systemctl disable dh360-backend.service 2>/dev/null || true
fi
EOF
chmod 755 "${BUILD_ROOT}/DEBIAN/prerm"

# 7. Build .deb package using dpkg-deb if available, or tar fallback
DEB_FILE="${DIST_DIR}/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
if command -v dpkg-deb >/dev/null 2>&1; then
    dpkg-deb --build "${BUILD_ROOT}" "${DEB_FILE}"
    echo "Created Debian package: ${DEB_FILE}"
else
    echo "dpkg-deb not found on host. Package structure prepared at: ${BUILD_ROOT}"
    echo "To build on Debian/Ubuntu/Raspberry Pi: dpkg-deb --build ${BUILD_ROOT} ${DEB_FILE}"
fi
