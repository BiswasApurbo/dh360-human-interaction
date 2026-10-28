# DH360 × Raspberry Pi 5 Deployment & Operational Release Guide

**Target Hardware**: Raspberry Pi 5 (8 GB RAM, Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4 GHz)  
**Operating System**: Raspberry Pi OS Bookworm (64-bit ARM64, Debian 12)  
**Target Throughput**: 18–25 FPS steady-state edge tracking  

---

## 1. System Requirements & Hardware Prerequisites

1. **Raspberry Pi 5 (8 GB recommended, 4 GB minimum)**.
2. **Raspberry Pi 27W USB-C Power Supply** (ensures stable current under peak CPU/camera load).
3. **Official Raspberry Pi Active Cooler** (mandatory: keeps sustained temperatures below 65°C to avoid thermal throttling).
4. **Camera Sensor**:
   - **USB UVC Webcam** (Plug-and-play at `/dev/video0` or `/dev/video1`).
   - **CSI Ribbon Camera** (Requires `libcamerify` wrapper if using libcamera stack).

---

## 2. Automated Installation & Setup

Clone the repository into `/home/pi/dh360-human-interaction`:

```bash
cd /home/pi
git clone https://github.com/BiswasApurbo/dh360-human-interaction.git
cd dh360-human-interaction
```

Run the automated installer:

```bash
chmod +x scripts/pi_setup.sh
./scripts/pi_setup.sh
```

The script automatically:
1. Installs headless system libraries (`libgl1`, `libglib2.0-0`, `v4l-utils`).
2. Creates `.venv` with Python 3.11+.
3. Installs package dependencies with NEON SIMD optimizations.
4. Downloads and validates YuNet and SFace ONNX model weights (`scripts/download_models.py`).

---

## 3. Configuration & Security Setup

1. Copy the environment template:
   ```bash
   cp .env.example .env
   chmod 600 .env
   ```
2. Configure your secret operator token:
   ```bash
   # Generate a 32-byte hex token
   openssl rand -hex 32
   ```
   Set this token in `.env`:
   ```bash
   DH360_HI_API__OPERATOR_TOKEN="your-generated-hex-token"
   ```
3. Set your external AI provider credentials in `.env` (kept strictly on the Pi, never in Git):
   ```bash
   GEMINI_API_KEY="AIzaSy..."
   SONIOX_API_KEY="your-soniox-key"
   GOOGLE_APPLICATION_CREDENTIALS="/home/pi/dh360-human-interaction/.credentials/tts-sa.json"
   ```

---

## 4. Production Service Management (`systemd`)

Install the systemd service to run DH360 as a background daemon on boot:

```bash
sudo cp scripts/systemd/dh360-hi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dh360-hi.service
sudo systemctl start dh360-hi.service
```

### Checking Status & Logs:
```bash
# Check service health
sudo systemctl status dh360-hi.service

# View live unbuffered logs
sudo journalctl -u dh360-hi.service -f
```

---

## 5. Bridging Raspberry Pi to Vercel Dashboard

Because Vercel runs in the cloud and cannot directly reach private LAN IP addresses (`192.168.x.x`), establish a secure tunnel to expose the FastAPI daemon (`127.0.0.1:8000`) without opening public router firewall ports:

### Recommended: Cloudflare Tunnel (Free & Authenticated)
1. Install `cloudflared` on the Pi:
   ```bash
   curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
   sudo dpkg -i cloudflared.deb
   ```
2. Start tunnel:
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
3. Note the assigned tunnel URL (e.g., `https://dh360-pi.yourdomain.com`).
4. Set this URL as `DH360_API_URL` in your Vercel Project Environment Variables.

The Next.js BFF server forwards authenticated dashboard requests through this tunnel, injecting the private `DH360_OPERATOR_TOKEN`. The browser client never touches the Pi directly.
