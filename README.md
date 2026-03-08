# Scriptorium

A collection of system administration and optimisation scripts, organised by platform.

## Scripts

### Windows

| Script | Description |
|--------|-------------|
| [`Optimise-Win11-OfficeVM.ps1`](windows/Optimise-Win11-OfficeVM.ps1) | Aggressively optimises a Windows 11 Pro VM for single-purpose Microsoft Office / Visio use. Removes bloat, disables telemetry, tunes services and UI. |

> **Warning:** The Windows optimisation script makes **irreversible changes** — it disables core services, removes built-in apps, and modifies system policies. It is designed exclusively for disposable, single-purpose VMs. **Do not run it on a daily-driver machine or any system you are not prepared to reinstall.** If you do not fully understand what it does, do not run it.

### Linux

| Script | Description |
|--------|-------------|
| [`Install-KDE-Plasma.sh`](linux/Install-KDE-Plasma.sh) | Installs KDE Plasma alongside GNOME on Ubuntu 25.10. Keeps GDM as the display manager, configures cross-toolkit theming, per-DE app associations, and autostart conflict prevention. |

## Usage

All shell scripts use a `#!/usr/bin/env bash` shebang. Run them with `./` (not `sh`) so the correct interpreter is used:

```bash
# Linux
chmod +x linux/Install-KDE-Plasma.sh
sudo ./linux/Install-KDE-Plasma.sh
```

```powershell
# Windows (run in an elevated PowerShell)
.\windows\Optimise-Win11-OfficeVM.ps1
```

> **Note:** Do not run `.sh` scripts with `sh script.sh` — they require Bash and will fail under other shells.
