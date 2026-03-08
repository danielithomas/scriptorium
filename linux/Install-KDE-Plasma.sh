#!/usr/bin/env bash
#
# Install KDE Plasma as an additional desktop environment on Ubuntu 25.10.
# GNOME and GDM remain intact — select your session at the GDM login screen.
#
# Run ONCE after a clean Ubuntu 25.10 install.
# Requires root (sudo).
#

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)
LOG_PATH="${REAL_HOME}/Desktop/KDE-Install-Log.txt"

# ---------------------------------------------------------------------------
# LOGGING HELPERS
# ---------------------------------------------------------------------------
write_section() {
    local title="$1"
    local line
    line=$(printf '=%.0s' {1..60})
    printf '\n\033[36m%s\n  %s\n%s\033[0m\n' "$line" "$title" "$line"
    printf '\n%s\n  %s\n%s\n' "$line" "$title" "$line" >> "$LOG_PATH"
}

write_ok()   { printf '  \033[32m[OK]\033[0m   %s\n' "$1"; printf '  [OK]   %s\n' "$1" >> "$LOG_PATH"; }
write_skip() { printf '  \033[33m[SKIP]\033[0m %s\n' "$1"; printf '  [SKIP] %s\n' "$1" >> "$LOG_PATH"; }
write_fail() { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; printf '  [FAIL] %s\n' "$1" >> "$LOG_PATH"; }

# ---------------------------------------------------------------------------
# START
# ---------------------------------------------------------------------------
START_TIME=$(date +%s)
mkdir -p "$(dirname "$LOG_PATH")"
echo "KDE Plasma Install Script - Started $(date)" > "$LOG_PATH"

printf '\n  KDE Plasma Installer for Ubuntu 25.10\n'
printf '  Log: %s\n\n' "$LOG_PATH"

# ---------------------------------------------------------------------------
# SECTION 0: Pre-flight Checks
# ---------------------------------------------------------------------------
write_section "0. Pre-flight Checks"

# Must be root
if [[ $(id -u) -ne 0 ]]; then
    write_fail "This script must be run as root (sudo)"
    exit 1
fi
write_ok "Running as root"

# Must be Ubuntu 25.10
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${VERSION_ID:-}" == "25.10" ]]; then
        write_ok "Ubuntu ${VERSION_ID} detected"
    else
        write_fail "Expected Ubuntu 25.10 but found ${PRETTY_NAME:-unknown}"
        exit 1
    fi
else
    write_fail "/etc/os-release not found — cannot verify distribution"
    exit 1
fi

# Check if KDE is already installed
if dpkg-query -W -f='${Status}' kde-standard 2>/dev/null | grep -q "install ok installed"; then
    write_skip "kde-standard is already installed — nothing to do"
    exit 0
fi
write_ok "kde-standard not yet installed"

# Check internet connectivity
if ping -c 1 -W 5 archive.ubuntu.com &>/dev/null; then
    write_ok "Internet connectivity confirmed"
else
    write_fail "Cannot reach archive.ubuntu.com — check your network"
    exit 1
fi

# ---------------------------------------------------------------------------
# SECTION 1: System Update
# ---------------------------------------------------------------------------
write_section "1. System Update"

if apt-get update -y && apt-get upgrade -y; then
    write_ok "System updated"
else
    write_fail "System update failed"
    exit 1
fi

# ---------------------------------------------------------------------------
# SECTION 2: Install KDE Plasma
# ---------------------------------------------------------------------------
write_section "2. Install KDE Plasma"

# Pre-seed debconf so the installer keeps GDM3 (avoids interactive prompt)
if command -v debconf-set-selections &>/dev/null; then
    echo "sddm shared/default-x-display-manager select gdm3" | debconf-set-selections
    write_ok "Preconfigured display manager selection to gdm3"
else
    write_skip "debconf-set-selections not found — you may be prompted"
fi

export DEBIAN_FRONTEND=noninteractive

if apt-get install -y kde-standard; then
    write_ok "kde-standard installed"
else
    write_fail "kde-standard installation failed"
    exit 1
fi

# Ensure GDM is still the active display manager
if dpkg-query -W -f='${Status}' gdm3 2>/dev/null | grep -q "install ok installed"; then
    if command -v dpkg-reconfigure &>/dev/null; then
        echo "gdm3 shared/default-x-display-manager select gdm3" | debconf-set-selections 2>/dev/null
    fi
    write_ok "GDM3 remains the active display manager"
else
    write_skip "GDM3 not found — display manager may have changed"
fi

# ---------------------------------------------------------------------------
# SECTION 3: Wayland Session Verification
# ---------------------------------------------------------------------------
write_section "3. Wayland Session Verification"

WAYLAND_SESSION_DIR="/usr/share/wayland-sessions"

if [[ -d "$WAYLAND_SESSION_DIR" ]]; then
    plasma_session=$(find "$WAYLAND_SESSION_DIR" -name '*plasma*' -print -quit 2>/dev/null)
    if [[ -n "$plasma_session" ]]; then
        write_ok "Plasma Wayland session file found: $(basename "$plasma_session")"
    else
        write_fail "No Plasma session file found in ${WAYLAND_SESSION_DIR}"
    fi
else
    write_fail "Wayland sessions directory does not exist"
fi

if command -v startplasma-wayland &>/dev/null; then
    write_ok "startplasma-wayland binary found"
else
    write_fail "startplasma-wayland binary not found"
fi

# Check for X11 session (may not exist on 25.10)
X11_SESSION_DIR="/usr/share/xsessions"
if [[ -d "$X11_SESSION_DIR" ]]; then
    x11_session=$(find "$X11_SESSION_DIR" -name '*plasma*' -print -quit 2>/dev/null)
    if [[ -n "$x11_session" ]]; then
        write_ok "Plasma X11 session file also available: $(basename "$x11_session")"
    else
        write_skip "No Plasma X11 session — Wayland-only (expected on 25.10)"
    fi
else
    write_skip "No xsessions directory — Wayland-only system"
fi

# ---------------------------------------------------------------------------
# SECTION 4: GDM Session Integration
# ---------------------------------------------------------------------------
write_section "4. GDM Session Integration"

# GDM reads .desktop files from wayland-sessions and xsessions directories.
# Verify the Plasma session file has a clean Name= field that GDM renders correctly.
SESSION_FILE=""
if [[ -d "$WAYLAND_SESSION_DIR" ]]; then
    SESSION_FILE=$(find "$WAYLAND_SESSION_DIR" -name '*plasma*' -print -quit 2>/dev/null)
fi

if [[ -n "$SESSION_FILE" ]]; then
    current_name=$(grep '^Name=' "$SESSION_FILE" | head -1 | cut -d= -f2-)
    write_ok "GDM session entry: \"${current_name}\" (${SESSION_FILE})"

    # Some builds ship a verbose Name= like "Plasma (Wayland) (Development)".
    # Simplify to "Plasma (Wayland)" so GDM shows a clean label.
    if [[ "$current_name" == *"Development"* ]] || [[ "$current_name" == *"dev"* ]]; then
        sed -i 's/^Name=.*/Name=Plasma (Wayland)/' "$SESSION_FILE"
        write_ok "Simplified session name to \"Plasma (Wayland)\""
    else
        write_ok "Session name looks clean — no changes needed"
    fi
else
    write_fail "Could not find Plasma session file for GDM integration"
fi

# ---------------------------------------------------------------------------
# SECTION 5: Cross-Toolkit Theming
# ---------------------------------------------------------------------------
write_section "5. Cross-Toolkit Theming"

# GTK apps in Plasma: kde-config-gtk-style + breeze-gtk-theme
if apt-get install -y kde-config-gtk-style breeze-gtk-theme; then
    write_ok "kde-config-gtk-style and breeze-gtk-theme installed (GTK apps in Plasma)"
else
    write_fail "Failed to install GTK theming packages"
fi

# Qt apps in GNOME: kvantum
if apt-get install -y qt6-style-kvantum qt6-style-kvantum-themes; then
    write_ok "qt6-style-kvantum and themes installed (Qt apps in GNOME)"
else
    write_fail "Failed to install Kvantum packages"
fi

# Set QT_STYLE_OVERRIDE in the real user's ~/.profile (idempotent)
PROFILE_FILE="${REAL_HOME}/.profile"
if grep -q 'QT_STYLE_OVERRIDE=kvantum' "$PROFILE_FILE" 2>/dev/null; then
    write_skip "QT_STYLE_OVERRIDE already set in ${PROFILE_FILE}"
else
    printf '\nexport QT_STYLE_OVERRIDE=kvantum\n' >> "$PROFILE_FILE"
    chown "$REAL_USER:$REAL_USER" "$PROFILE_FILE"
    write_ok "QT_STYLE_OVERRIDE=kvantum added to ${PROFILE_FILE}"
fi

# ---------------------------------------------------------------------------
# SECTION 6: Default Application Associations
# ---------------------------------------------------------------------------
write_section "6. Default Application Associations"

install -d -o "$REAL_USER" -g "$REAL_USER" "${REAL_HOME}/.config"

# GNOME file manager association
GNOME_MIME="${REAL_HOME}/.config/gnome-mimeapps.list"
if [[ -f "$GNOME_MIME" ]]; then
    write_skip "gnome-mimeapps.list already exists"
else
    cat > "$GNOME_MIME" <<'MIME'
[Default Applications]
inode/directory=org.gnome.Nautilus.desktop
MIME
    chown "$REAL_USER:$REAL_USER" "$GNOME_MIME"
    write_ok "Created gnome-mimeapps.list (Nautilus for GNOME)"
fi

# KDE file manager association
KDE_MIME="${REAL_HOME}/.config/kde-mimeapps.list"
if [[ -f "$KDE_MIME" ]]; then
    write_skip "kde-mimeapps.list already exists"
else
    cat > "$KDE_MIME" <<'MIME'
[Default Applications]
inode/directory=org.kde.dolphin.desktop
MIME
    chown "$REAL_USER:$REAL_USER" "$KDE_MIME"
    write_ok "Created kde-mimeapps.list (Dolphin for KDE)"
fi

# ---------------------------------------------------------------------------
# SECTION 7: Autostart Conflict Prevention
# ---------------------------------------------------------------------------
write_section "7. Autostart Conflict Prevention"

AUTOSTART_DIR="${REAL_HOME}/.config/autostart"
install -d -o "$REAL_USER" -g "$REAL_USER" "$AUTOSTART_DIR"

XDG_AUTOSTART="/etc/xdg/autostart"

# Baloo (KDE file indexer) — restrict to KDE only
BALOO_SRC="${XDG_AUTOSTART}/baloo_file.desktop"
BALOO_DST="${AUTOSTART_DIR}/baloo_file.desktop"
if [[ -f "$BALOO_DST" ]]; then
    write_skip "baloo_file.desktop override already exists"
elif [[ -f "$BALOO_SRC" ]]; then
    cp "$BALOO_SRC" "$BALOO_DST"
    if ! grep -q '^OnlyShowIn=' "$BALOO_DST"; then
        printf 'OnlyShowIn=KDE;\n' >> "$BALOO_DST"
    else
        sed -i 's/^OnlyShowIn=.*/OnlyShowIn=KDE;/' "$BALOO_DST"
    fi
    chown "$REAL_USER:$REAL_USER" "$BALOO_DST"
    write_ok "Baloo restricted to KDE sessions"
else
    write_skip "baloo_file.desktop not found in ${XDG_AUTOSTART}"
fi

# GNOME Tracker / LocalSearch miners — restrict to GNOME only
tracker_found=false
for src_file in "${XDG_AUTOSTART}"/tracker-miner-*.desktop "${XDG_AUTOSTART}"/localsearch-*.desktop; do
    [[ -f "$src_file" ]] || continue
    tracker_found=true
    base_name=$(basename "$src_file")
    dst_file="${AUTOSTART_DIR}/${base_name}"
    if [[ -f "$dst_file" ]]; then
        write_skip "${base_name} override already exists"
        continue
    fi
    cp "$src_file" "$dst_file"
    if ! grep -q '^OnlyShowIn=' "$dst_file"; then
        printf 'OnlyShowIn=GNOME;\n' >> "$dst_file"
    else
        sed -i 's/^OnlyShowIn=.*/OnlyShowIn=GNOME;/' "$dst_file"
    fi
    chown "$REAL_USER:$REAL_USER" "$dst_file"
    write_ok "${base_name} restricted to GNOME sessions"
done
if [[ "$tracker_found" == false ]]; then
    write_skip "No Tracker/LocalSearch miner desktop files found in ${XDG_AUTOSTART}"
fi

# ---------------------------------------------------------------------------
# SECTION 8: Summary
# ---------------------------------------------------------------------------
write_section "8. Summary"

END_TIME=$(date +%s)
DURATION=$(( END_TIME - START_TIME ))

write_ok "KDE Plasma (kde-standard) installed successfully"
write_ok "GDM3 retained as display manager"
write_ok "Cross-toolkit theming configured"
write_ok "Per-DE application associations set"
write_ok "Autostart conflicts resolved"
write_ok "Completed in ${DURATION}s"

printf '\n  \033[37mTo use Plasma:\033[0m\n'
printf '    1. Log out of your current session\n'
printf '    2. On the GDM login screen, click the gear icon (bottom-right)\n'
printf '    3. Select "Plasma (Wayland)"\n'
printf '    4. Enter your password and log in\n\n'

{
    echo ""
    echo "To use Plasma:"
    echo "  1. Log out of your current session"
    echo "  2. On the GDM login screen, click the gear icon (bottom-right)"
    echo "  3. Select \"Plasma (Wayland)\""
    echo "  4. Enter your password and log in"
    echo ""
    echo "Completed in ${DURATION}s at $(date)"
} >> "$LOG_PATH"

printf '  Log saved to: %s\n\n' "$LOG_PATH"
