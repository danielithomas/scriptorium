# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scriptorium is a collection of system optimization and administration scripts, organized by platform. Currently contains PowerShell scripts for Windows VM optimization, with a `linux/` directory planned for future expansion.

## Repository Structure

```
scriptorium/
├── windows/    # PowerShell scripts for Windows optimization
├── linux/      # (empty) Future Linux scripts
```

## Script Conventions

### PowerShell Scripts (windows/)

- Scripts use `#Requires -RunAsAdministrator` for elevated operations
- Use `Set-StrictMode -Version Latest` at the top of scripts
- Helper functions follow a consistent pattern: `Write-Section`, `Write-OK`, `Write-SKIP`, `Write-FAIL` for structured logging with color-coded console output and file logging
- Registry changes use the `Set-RegValue` wrapper (auto-creates parent keys, handles errors)
- Service changes use the `Disable-Service` wrapper (stops then disables, handles missing services)
- All destructive operations are wrapped in try-catch with status reporting
- Scripts create system restore points before making changes
- Output goes to both console (color-coded) and log files on Desktop

### Adding New Scripts

- Place platform-specific scripts in their respective directory (`windows/`, `linux/`)
- Follow the section-based structure with numbered sections and `Write-Section` headers
- Use the established helper functions for consistent logging and error handling
- Wrap operations in try-catch to report success/skip/fail rather than crashing

## Git Workflow

- `main` is the default branch
- `dev` branch for development
