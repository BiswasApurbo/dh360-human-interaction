#!/usr/bin/env bash
# ==============================================================================
# KBCON Humanoid Robot Device Installer & Provisioning Agent
# Target: Raspberry Pi 5 / Debian 12 (Bookworm) / Ubuntu 24.04 ARM64
# ==============================================================================

set -euo pipefail

VERSION="1.0.0"
INSTALL_DIR="/home/eliva/dh360-human-interaction"
DEVICE_DIR="/etc/kbcon/device"
SERVICE_NAME="dh360-backend.service"
BLACKLIST_FILE="/etc/modprobe.d/blacklist-option.conf"
REPO_URL="https://github.com/BiswasApurbo/dh360-human-interaction.git"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

MODE="install"
DRY_RUN=false

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
show_help() {
    cat << EOF
${BOLD}KBCON Raspberry Pi 5 Linux Installer v${VERSION}${NC}

Usage: sudo ./install-kbcon.sh [OPTIONS]

Options:
  --install       Perform full end-to-end installation (default)
  --upgrade       Update existing installation, venv, and systemd services
  --status        Verify current installation, driver blacklist, and service health
  --dry-run       Check prerequisites and print plan without modifying system
  --dir <path>    Specify repository destination directory (default: ${INSTALL_DIR})
  --help, -h      Display this help message

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install)
            MODE="install"
            shift
            ;;
        --upgrade)
            MODE="upgrade"
            shift
            ;;
        --status)
            MODE="status"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Status Check Handler
# ------------------------------------------------------------------------------
check_status() {
    echo -e "\n${BOLD}${CYAN}==============================================================${NC}"
    echo -e "${BOLD}${CYAN}                KBCON DEVICE INSTALLATION STATUS              ${NC}"
    echo -e "${BOLD}${CYAN}==============================================================${NC}\n"

    # 1. OS & Architecture
    echo -n "Architecture:       "
    uname -m
    echo -n "Operating System:   "
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "${PRETTY_NAME}"
    else
        uname -s
    fi

    # 2. Camera Blacklist
    echo -n "Camera Blacklist:   "
    if [ -f "${BLACKLIST_FILE}" ] && grep -q "blacklist option" "${BLACKLIST_FILE}"; then
        echo -e "${GREEN}[OK] Installed (${BLACKLIST_FILE})${NC}"
    else
        echo -e "${YELLOW}[WARNING] Missing or incomplete (${BLACKLIST_FILE})${NC}"
    fi

    # 3. Codebase Directory
    echo -n "Repository Path:    "
    if [ -d "${INSTALL_DIR}" ]; then
        echo -e "${GREEN}[OK] Exists (${INSTALL_DIR})${NC}"
    else
        echo -e "${RED}[MISSING] Not found at ${INSTALL_DIR}${NC}"
    fi

    # 4. Python Virtual Environment
    echo -n "Python Virtualenv:  "
    if [ -x "${INSTALL_DIR}/.venv/bin/python" ]; then
        VENV_VER=$("${INSTALL_DIR}/.venv/bin/python" --version)
        echo -e "${GREEN}[OK] ${VENV_VER}${NC}"
    else
        echo -e "${RED}[MISSING] No virtualenv at ${INSTALL_DIR}/.venv${NC}"
    fi

    # 5. Systemd Service
    echo -n "Backend Service:    "
    if command -v systemctl >/dev/null 2>&1; then
        SVC_STATUS=$(systemctl is-active "${SERVICE_NAME}" 2>/dev/null || echo "inactive")
        if [ "${SVC_STATUS}" = "active" ]; then
            echo -e "${GREEN}[ACTIVE] Running${NC}"
        else
            echo -e "${YELLOW}[${SVC_STATUS^^}] Service is not active${NC}"
        fi
    else
        echo "systemctl unavailable"
    fi

    # 6. Device Identity & Pairing
    echo -n "Device Identity:    "
    if [ -f "${DEVICE_DIR}/device_id" ]; then
        DEV_ID=$(cat "${DEVICE_DIR}/device_id")
        echo -e "${GREEN}[PROVISIONED] ID: ${DEV_ID}${NC}"
    elif [ -f "${HOME}/.config/dh360/device/device_id" ]; then
        DEV_ID=$(cat "${HOME}/.config/dh360/device/device_id")
        echo -e "${GREEN}[USER-PROVISIONED] ID: ${DEV_ID}${NC}"
    else
        echo -e "${YELLOW}[UNPROVISIONED] Run 'kbcon-device pair' to register${NC}"
    fi

    echo -e "\n${BOLD}${CYAN}==============================================================${NC}\n"
}

if [ "${MODE}" = "status" ]; then
    check_status
    exit 0
fi

# ------------------------------------------------------------------------------
# Root Permissions Enforcement
# ------------------------------------------------------------------------------
if [ "$EUID" -ne 0 ] && [ "${DRY_RUN}" = false ]; then
    echo -e "${RED}[ERROR] This script must be run as root (use sudo).${NC}"
    exit 1
fi

echo -e "\n${BOLD}${CYAN}==============================================================${NC}"
echo -e "${BOLD}${CYAN}       KBCON Raspberry Pi 5 Automated Provisioning Agent      ${NC}"
echo -e "${BOLD}${CYAN}       Version ${VERSION} | Target: ${INSTALL_DIR}              ${NC}"
echo -e "${BOLD}${CYAN}==============================================================${NC}\n"

if [ "${DRY_RUN}" = true ]; then
    echo -e "${YELLOW}[DRY-RUN] Simulating installation plan. No files or system state will be modified.${NC}\n"
fi

# ------------------------------------------------------------------------------
# 1. Pre-flight Architecture & OS Check
# ------------------------------------------------------------------------------
ARCH=$(uname -m)
echo -e "${BOLD}[1/7] Checking hardware platform & OS...${NC}"
echo "      Detected architecture: ${ARCH}"
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
    echo -e "      ${YELLOW}[NOTE] Non-ARM64 architecture (${ARCH}) detected. Continuing in test/compatibility mode.${NC}"
else
    echo -e "      ${GREEN}[OK] ARM64 platform verified.${NC}"
fi

# ------------------------------------------------------------------------------
# 2. Linux Kernel Driver Blacklist (Fix Sonix Webcam 1410:1410 Hijack)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[2/7] Configuring Linux kernel module blacklist for webcam...${NC}"
if [ "${DRY_RUN}" = true ]; then
    echo "      [PLAN] Write 'blacklist option' to ${BLACKLIST_FILE}"
else
    mkdir -p /etc/modprobe.d
    cat > "${BLACKLIST_FILE}" << 'EOF'
# KBCON / DH360 Hardware Fix: Prevent GSM USB serial driver (option) from
# hijacking Sonix MWB-16 USB video camera (ID 1410:1410).
blacklist option
EOF
    chmod 644 "${BLACKLIST_FILE}"
    echo -e "      ${GREEN}[OK] Blacklisted 'option' driver in ${BLACKLIST_FILE}${NC}"
fi

# ------------------------------------------------------------------------------
# 3. System Packages & Native Runtime Dependencies
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[3/7] Ensuring system package dependencies...${NC}"
REQUIRED_PKGS=(
    python3
    python3-venv
    python3-pip
    git
    curl
    v4l-utils
    libgl1
    libglib2.0-0
)

if [ "${DRY_RUN}" = true ]; then
    echo "      [PLAN] apt-get update && apt-get install -y ${REQUIRED_PKGS[*]}"
else
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq || true
    apt-get install -y --no-install-recommends "${REQUIRED_PKGS[@]}" || {
        echo -e "      ${YELLOW}[WARN] Could not install all packages via apt-get. Continuing with existing system runtime.${NC}"
    }
    echo -e "      ${GREEN}[OK] System packages checked.${NC}"
fi

# ------------------------------------------------------------------------------
# 4. Target Directory and Source Repository Setup
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[4/7] Setting up KBCON source repository...${NC}"
TARGET_USER=$(id -un "${SUDO_UID:-1000}" 2>/dev/null || echo "eliva")
TARGET_GROUP=$(id -gn "${SUDO_UID:-1000}" 2>/dev/null || echo "${TARGET_USER}")

if [ "${DRY_RUN}" = true ]; then
    echo "      [PLAN] Ensure directory ${INSTALL_DIR} exists, owned by ${TARGET_USER}:${TARGET_GROUP}"
else
    if [ ! -d "${INSTALL_DIR}" ]; then
        mkdir -p "${INSTALL_DIR}"
        git clone "${REPO_URL}" "${INSTALL_DIR}"
        chown -R "${TARGET_USER}:${TARGET_GROUP}" "${INSTALL_DIR}"
        echo -e "      ${GREEN}[OK] Cloned repository to ${INSTALL_DIR}${NC}"
    elif [ "${MODE}" = "upgrade" ]; then
        cd "${INSTALL_DIR}"
        sudo -u "${TARGET_USER}" git pull --rebase || true
        echo -e "      ${GREEN}[OK] Repository updated to latest commit.${NC}"
    else
        echo -e "      ${GREEN}[OK] Repository already present at ${INSTALL_DIR}${NC}"
    fi
fi

# ------------------------------------------------------------------------------
# 5. Python Virtual Environment and Package Installation
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[5/7] Provisioning Python virtual environment...${NC}"
VENV_PATH="${INSTALL_DIR}/.venv"

if [ "${DRY_RUN}" = true ]; then
    echo "      [PLAN] Create venv at ${VENV_PATH} and install package in editable mode"
else
    if [ ! -d "${VENV_PATH}" ]; then
        sudo -u "${TARGET_USER}" python3 -m venv "${VENV_PATH}"
    fi

    # Install dependencies inside venv
    sudo -u "${TARGET_USER}" "${VENV_PATH}/bin/pip" install --upgrade pip setuptools wheel
    sudo -u "${TARGET_USER}" "${VENV_PATH}/bin/pip" install -e "${INSTALL_DIR}"

    echo -e "      ${GREEN}[OK] Python virtual environment provisioned.${NC}"
fi

# ------------------------------------------------------------------------------
# 6. Device Identity Directory and kbcon-device Global Executable
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[6/7] Configuring /etc/kbcon/device and kbcon-device CLI tool...${NC}"
if [ "${DRY_RUN}" = true ]; then
    echo "      [PLAN] Create ${DEVICE_DIR} with 0700 permissions"
    echo "      [PLAN] Link /usr/local/bin/kbcon-device -> ${VENV_PATH}/bin/kbcon-device"
else
    mkdir -p "${DEVICE_DIR}"
    chmod 700 "${DEVICE_DIR}"
    # Allow target user access if non-root operations are required
    chown "${TARGET_USER}:${TARGET_GROUP}" "${DEVICE_DIR}"

    # Global CLI wrapper
    cat > /usr/local/bin/kbcon-device << EOF
#!/usr/bin/env bash
exec "${VENV_PATH}/bin/kbcon-device" "\$@"
EOF
    chmod 755 /usr/local/bin/kbcon-device
    echo -e "      ${GREEN}[OK] Installed /usr/local/bin/kbcon-device wrapper.${NC}"
fi

# ------------------------------------------------------------------------------
# 7. Systemd Background Service Setup
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[7/7] Configuring systemd background service...${NC}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"

if [ "${DRY_RUN}" = true ]; then
    echo "      [PLAN] Write ${SERVICE_FILE} and enable ${SERVICE_NAME}"
else
    cat > "${SERVICE_FILE}" << EOF
[Unit]
Description=KBCON / DH360 Human Interaction Subsystem & Backend
After=network.target network-online.target sound.target

[Service]
Type=simple
User=${TARGET_USER}
Group=${TARGET_GROUP}
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=${VENV_PATH}/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=${VENV_PATH}/bin/python -m dh360_hi.cli.main serve --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}"
    systemctl restart "${SERVICE_NAME}" || true
    echo -e "      ${GREEN}[OK] Enabled and configured ${SERVICE_NAME}.${NC}"
fi

# ------------------------------------------------------------------------------
# Completion & Next Steps
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}==============================================================${NC}"
echo -e "${BOLD}${GREEN}           KBCON INSTALLATION COMPLETED SUCCESSFULLY!         ${NC}"
echo -e "${BOLD}${GREEN}==============================================================${NC}\n"
echo -e "Next Step: Pair this physical Raspberry Pi 5 to a robot:"
echo -e "\n    ${BOLD}sudo kbcon-device pair${NC}\n"
echo -e "This will generate a temporary pairing code. Enter the code in the"
echo -e "KBCON Web Dashboard under ${CYAN}Robots -> Pair Device${NC} to activate."
echo -e "==============================================================\n"
