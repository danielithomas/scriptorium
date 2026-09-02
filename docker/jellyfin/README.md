# Jellyfin Media Server

Self-hosted media server with automatic subtitle management. Designed for Intel hosts with iGPU hardware transcoding (QSV/VAAPI).

## Services

| Service | Purpose | Port |
|---------|---------|------|
| **Jellyfin** | Media server — streaming, transcoding, metadata, client apps | 8096 (8920 HTTPS) |
| **Bazarr** | Automatic subtitle downloads (OpenSubtitles, Subscene, etc.) | 6767 |

Jellyfin also binds two fixed UDP ports on the host: **7359** for LAN client auto-discovery and
**1900** for DLNA (used as a fallback by some TVs). Both are unconditional — if another DLNA server
is already running on the host, that port will collide.

Bazarr starts only after Jellyfin reports healthy (`depends_on: service_healthy`), so a failing
Jellyfin healthcheck leaves Bazarr in `created` state rather than starting it.

## Quick Start

### 1. Create Media Directories

```bash
sudo mkdir -p /data/media/movies
sudo mkdir -p /data/media/tv
# Optional: split movies into subdirectories (by genre, language, era, …)
# and add each as its own Jellyfin library
# sudo mkdir -p /data/media/movies/{features,documentaries,kids}
sudo mkdir -p /data/docker/jellyfin/{config,cache,bazarr}
sudo chown -R 1000:1000 /data/media /data/docker/jellyfin
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set your media path, timezone, and server URL
```

### 3. Deploy

```bash
docker compose up -d
```

### 4. Initial Setup

1. Open `http://<host>:8096` in a browser
2. Follow the Jellyfin setup wizard (language, admin user, libraries)
3. Add libraries pointing to `/media/movies` and `/media/tv` (or to individual subdirectories if you split them)
4. Enable hardware transcoding: Dashboard → Playback → Transcoding → VAAPI or QSV
5. Create user accounts (admin + family members)

### 5. Bazarr Setup

1. Open `http://<host>:6767`
2. Configure subtitle providers (OpenSubtitles account recommended)
3. Connect to Jellyfin: Settings → Jellyfin → API key from Jellyfin dashboard
4. Set language profiles for the subtitle languages you want

## Hardware Transcoding (Intel)

The container mounts `/dev/dri` for Intel iGPU access. To enable:

1. Verify iGPU is available:
   ```bash
   ls -la /dev/dri/
   # Should show renderD128
   ```

2. Check render group:
   ```bash
   getent group render
   # Note the GID — set RENDER_GROUP in .env if not "render"
   ```

3. In Jellyfin Dashboard → Playback → Transcoding:
   - Hardware acceleration: **Intel QuickSync (QSV)** or **VAAPI**
   - Enable H.264 and HEVC hardware decoding
   - Enable hardware encoding

## Media Naming Convention

Follow Jellyfin's expected format for automatic metadata matching:

```
movies/
├── Example Movie (2019)/
│   └── Example Movie (2019).mp4
├── Another Film (2021)/
│   ├── Another Film (2021).mp4
│   └── Another Film (2021).srt
└── Third Title (2023)/
    └── Third Title (2023).mkv
```

**Key rules:**
- One folder per movie: `Movie Name (Year)/`
- Video file matches folder: `Movie Name (Year).ext`
- Subtitles alongside video: `.srt`, `.ass`, `.sub`
- Jellyfin auto-scrapes metadata from TMDB

## Client Apps

| Platform | App | Notes |
|----------|-----|-------|
| LG TV (WebOS 6+) | Jellyfin (LG Content Store) | Direct LAN, remote control |
| iOS | Jellyfin / Swiftfin | Swiftfin has better UI |
| Android | Jellyfin / Findroid | Material Design |
| Web | `http://<host>:8096` | Any browser |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_PATH` | `/data/docker/jellyfin` | Persistent config storage |
| `MEDIA_PATH` | `/data/media` | Root media directory |
| `JELLYFIN_PORT` | `8096` | Web UI / API port |
| `JELLYFIN_HTTPS_PORT` | `8920` | HTTPS port (optional — only used if TLS is configured in Jellyfin) |
| `BAZARR_PORT` | `6767` | Bazarr web UI |
| `TZ` | `UTC` | Timezone (IANA name, e.g. `Europe/London`) |
| `PUID` / `PGID` | `1000` | File ownership user/group |
| `RENDER_GROUP` | `render` | GPU access group name |
| `SERVER_URL` | `http://localhost:8096` | Published URL for clients |

## Volumes

| Mount | Purpose |
|-------|---------|
| `/config` | Jellyfin database, metadata, plugins |
| `/cache` | Transcoding cache, image cache |
| `/media` | Media library (movies, TV) |

## Resource Usage

- **RAM:** ~200-500 MB (idle), up to 2 GB during transcoding
- **CPU:** Minimal with hardware transcoding; CPU fallback uses 2-4 cores
- **Disk:** Config ~500 MB; cache varies with transcoding activity
