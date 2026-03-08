#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Optimise Windows 11 Pro VM for Microsoft Office / Visio only use.
    Designed for VMware Workstation guests.

.DESCRIPTION
    - Removes AppX bloat
    - Disables unnecessary services
    - Kills telemetry and scheduled tasks
    - Applies performance and visual tweaks
    - Tunes Office after installation
    - Produces a summary log

    !! WARNING !!
    This script makes AGGRESSIVE, IRREVERSIBLE changes to your Windows installation.
    It disables core services, removes built-in apps, alters boot configuration, and
    modifies system policies. It is designed EXCLUSIVELY for single-purpose VMs running
    Microsoft Office. DO NOT run this on your daily-driver machine, a domain-joined PC,
    or any system you are not prepared to reinstall from scratch.

    If you do not fully understand what every section of this script does, DO NOT RUN IT.

.NOTES
    Run ONCE after a clean Windows 11 Pro install with VMware Tools installed.
    Some changes require a reboot to take full effect.
    A system restore point is created automatically before any changes.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "SilentlyContinue"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
$LogPath    = "$env:USERPROFILE\Desktop\VM-Optimise-Log.txt"
$Transcript = "$env:USERPROFILE\Desktop\VM-Optimise-Transcript.txt"

# ---------------------------------------------------------------------------
# LOGGING HELPERS
# ---------------------------------------------------------------------------
function Write-Section {
    param([string]$Title)
    $line = "=" * 60
    Write-Host "`n$line" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "$line" -ForegroundColor Cyan
    Add-Content $LogPath "`n$line`n  $Title`n$line"
}

function Write-OK   { param([string]$Msg) Write-Host "  [OK]   $Msg" -ForegroundColor Green;  Add-Content $LogPath "  [OK]   $Msg" }
function Write-SKIP { param([string]$Msg) Write-Host "  [SKIP] $Msg" -ForegroundColor Yellow; Add-Content $LogPath "  [SKIP] $Msg" }
function Write-FAIL { param([string]$Msg) Write-Host "  [FAIL] $Msg" -ForegroundColor Red;    Add-Content $LogPath "  [FAIL] $Msg" }

function Disable-Service {
    param([string]$Name)
    try {
        $svc = Get-Service -Name $Name -ErrorAction Stop
        Stop-Service -Name $Name -Force -ErrorAction SilentlyContinue
        Set-Service  -Name $Name -StartupType Disabled
        Write-OK "Service disabled: $Name"
    } catch {
        Write-SKIP "Service not found: $Name"
    }
}

function Set-RegValue {
    param([string]$Path, [string]$Name, $Value, [string]$Type = "DWORD")
    try {
        if (-not (Test-Path $Path)) { New-Item -Path $Path -Force | Out-Null }
        Set-ItemProperty -Path $Path -Name $Name -Value $Value -Type $Type -Force
        Write-OK "Registry: $Path\$Name = $Value"
    } catch {
        Write-FAIL "Registry failed: $Path\$Name"
    }
}

# ---------------------------------------------------------------------------
# START
# ---------------------------------------------------------------------------
Start-Transcript -Path $Transcript -Force
$StartTime = Get-Date
Add-Content $LogPath "VM Optimisation Script - Started $StartTime"

Write-Host "`n  Windows 11 Pro - Office/Visio VM Optimiser" -ForegroundColor White
Write-Host "  Log: $LogPath`n" -ForegroundColor Gray

# ---------------------------------------------------------------------------
# SECTION 0: Restore Point
# ---------------------------------------------------------------------------
Write-Section "0. Creating System Restore Point"
try {
    Enable-ComputerRestore -Drive "C:\" -ErrorAction Stop
    Checkpoint-Computer -Description "Pre-VM-Optimise" -RestorePointType "MODIFY_SETTINGS"
    Write-OK "Restore point created"
} catch {
    Write-SKIP "Could not create restore point (may already exist or VSS disabled)"
}

# ---------------------------------------------------------------------------
# SECTION 1: Boot Optimisation
# ---------------------------------------------------------------------------
Write-Section "1. Boot Optimisation"

bcdedit /set quietboot yes        | Out-Null ; Write-OK "Quiet boot enabled"
bcdedit /set nointegritychecks on | Out-Null ; Write-OK "Integrity checks disabled"

# Reduce boot menu timeout
bcdedit /timeout 3 | Out-Null ; Write-OK "Boot timeout set to 3s"

# Disable Fast Startup (causes VM resume issues)
Set-RegValue "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power" "HiberbootEnabled" 0

# Disable hibernate entirely (reclaims hiberfil.sys disk space)
powercfg -h off | Out-Null ; Write-OK "Hibernation disabled"

# Apply Ultimate Performance power plan
$planGuid = "e9a42b02-d5df-448d-aa00-03f14749eb61"
powercfg -duplicatescheme $planGuid 2>$null | Out-Null
$newPlan = powercfg -list | Select-String $planGuid
if ($newPlan) {
    $newGuid = ($newPlan -split "\s+")[3]
    powercfg -setactive $newGuid | Out-Null
    Write-OK "Ultimate Performance power plan activated"
} else {
    # Fall back to High Performance
    powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c | Out-Null
    Write-OK "High Performance power plan activated (fallback)"
}

# ---------------------------------------------------------------------------
# SECTION 2: Visual Effects - Best Performance (keep font smoothing)
# ---------------------------------------------------------------------------
Write-Section "2. Visual Effects"

Set-RegValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\VisualEffects" "VisualFXSetting" 2

# Explicitly set individual effect flags
$vfxPath = "HKCU:\Control Panel\Desktop"
Set-RegValue $vfxPath "DragFullWindows"         "0"  String
Set-RegValue $vfxPath "MenuShowDelay"           "0"  String
Set-RegValue $vfxPath "UserPreferencesMask"     ([byte[]](0x90,0x12,0x01,0x80)) Binary
Set-RegValue $vfxPath "FontSmoothing"           "2"  String  # Keep - text is unreadable without it
Set-RegValue $vfxPath "FontSmoothingType"       2

$winMetrics = "HKCU:\Control Panel\Desktop\WindowMetrics"
Set-RegValue $winMetrics "MinAnimate" "0" String

# ---------------------------------------------------------------------------
# SECTION 3: Memory Tweaks
# ---------------------------------------------------------------------------
Write-Section "3. Memory Tweaks"

# Disable pagefile (6GB fixed RAM is plenty for Office/Visio)
try {
    $cs = Get-WmiObject -Class Win32_ComputerSystem
    $cs.AutomaticManagedPagefile = $false
    $cs.Put() | Out-Null
    $pf = Get-WmiObject -Class Win32_PageFileSetting
    if ($pf) { $pf.Delete() | Out-Null }
    Write-OK "Pagefile disabled"
} catch {
    Write-FAIL "Could not disable pagefile via WMI"
}

# Disable memory compression (less overhead, fine for dedicated VM)
try {
    Disable-MMAgent -MemoryCompression -ErrorAction Stop
    Write-OK "Memory compression disabled"
} catch {
    Write-SKIP "Memory compression already disabled or not supported"
}

# Keep frequently used apps in RAM (disable RAM trimming)
Set-RegValue "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" "DisablePagingExecutive" 1
Set-RegValue "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" "LargeSystemCache" 0

# ---------------------------------------------------------------------------
# SECTION 4: Telemetry and Privacy
# ---------------------------------------------------------------------------
Write-Section "4. Telemetry and Privacy"

Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\DataCollection"   "AllowTelemetry"                   0
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\DataCollection"   "DisableOneSettingsDownloads"      1
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\DataCollection"   "DoNotShowFeedbackNotifications"   1
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\Windows Search"   "AllowCortana"                     0
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\CloudContent"     "DisableWindowsConsumerFeatures"   1
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\CloudContent"     "DisableSoftLanding"               1
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\System"           "EnableActivityFeed"               0
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\System"           "PublishUserActivities"            0
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\OneDrive"         "DisableFileSyncNGSC"              1
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\OneDrive"         "PreventNetworkTrafficPreUserSignIn" 1
Set-RegValue "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\Explorer" "NoOneDriveSync" 1

# Disable advertising ID
Set-RegValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\AdvertisingInfo" "Enabled" 0

# Disable app launch tracking
Set-RegValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced" "Start_TrackProgs" 0

# Disable suggested content in Settings
Set-RegValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager" "SubscribedContent-338393Enabled" 0
Set-RegValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager" "SubscribedContent-353694Enabled" 0
Set-RegValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager" "SubscribedContent-353696Enabled" 0
Set-RegValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager" "SystemPaneSuggestionsEnabled"    0
Set-RegValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager" "SoftLandingEnabled"              0

# ---------------------------------------------------------------------------
# SECTION 5: Disable Services
# ---------------------------------------------------------------------------
Write-Section "5. Services"

$services = @(
    # Xbox
    "XblAuthManager", "XblGameSave", "XboxGipSvc", "XboxNetApiSvc",
    # Search / Indexing
    "WSearch",
    # Superfetch (pointless in VM)
    "SysMain",
    # Telemetry
    "DiagTrack", "dmwappushservice", "DPS", "WdiServiceHost", "WdiSystemHost",
    # Remote access (disable if you don't RDP into this VM)
    "SessionEnv", "TermService", "UmRdpService", "RemoteRegistry",
    # Hardware not present in VM
    "bthserv", "WbioSrvc", "SensorDataService", "SensorService",
    "SensrSvc", "WiaWvc", "stisvc",
    # Networking junk
    "lmhosts", "IKEEXT", "PolicyAgent", "SharedAccess", "icssvc",
    # Misc
    "Fax", "MapsBroker", "lfsvc", "WMPNetworkSvc", "PhoneSvc",
    "TapiSrv", "WerSvc", "wercplsupport", "RetailDemo",
    "wisvc",        # Insider service
    "OneSyncSvc",   # Sync
    "MixedRealityOpenXRSvc",
    "spectrum",     # Spatial sound
    "TabletInputService",
    "WpnService",   # Push notifications
    "WpnUserService"
)

foreach ($svc in $services) { Disable-Service $svc }

# ---------------------------------------------------------------------------
# SECTION 6: Remove AppX Bloat
# ---------------------------------------------------------------------------
Write-Section "6. Removing AppX Packages"

# Packages to KEEP (Office needs some of these runtimes)
$keepList = @(
    "Microsoft.WindowsStore",
    "Microsoft.DesktopAppInstaller",
    "Microsoft.VCLibs",
    "Microsoft.UI.Xaml",
    "Microsoft.NET",
    "Microsoft.WindowsTerminal",
    "Microsoft.MicrosoftEdge",           # Keep - Office links open here
    "Microsoft.AAD.BrokerPlugin",        # Keep - M365 auth
    "Microsoft.Windows.CloudExperienceHost",
    "Microsoft.Windows.ShellExperienceHost",
    "Microsoft.WindowsCalculator"        # Optional - lightweight, sometimes useful
)

$keepPattern = $keepList -join "|"

# Remove provisioned packages (applies to new users too)
Get-AppxProvisionedPackage -Online |
    Where-Object { $_.PackageName -notmatch $keepPattern } |
    ForEach-Object {
        try {
            Remove-AppxProvisionedPackage -Online -PackageName $_.PackageName -ErrorAction Stop | Out-Null
            Write-OK "Removed provisioned: $($_.DisplayName)"
        } catch {
            Write-FAIL "Could not remove provisioned: $($_.DisplayName)"
        }
    }

# Remove for current user
Get-AppxPackage |
    Where-Object { $_.Name -notmatch $keepPattern } |
    ForEach-Object {
        try {
            Remove-AppxPackage -Package $_.PackageFullName -ErrorAction Stop | Out-Null
            Write-OK "Removed package: $($_.Name)"
        } catch {
            Write-FAIL "Could not remove: $($_.Name)"
        }
    }

# ---------------------------------------------------------------------------
# SECTION 7: Startup Items
# ---------------------------------------------------------------------------
Write-Section "7. Startup Items"

$startupKeys = @(
    "OneDrive",
    "OneDriveSetup",
    "com.squirrel.Teams.Teams",
    "Teams",
    "MicrosoftEdgeAutoLaunch",
    "Skype",
    "Discord"
)

foreach ($key in $startupKeys) {
    try {
        Remove-ItemProperty -Path "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run" -Name $key -Force -ErrorAction Stop
        Write-OK "Removed startup: $key"
    } catch {
        Write-SKIP "Startup item not present: $key"
    }
    try {
        Remove-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run" -Name $key -Force -ErrorAction Stop
        Write-OK "Removed startup (HKLM): $key"
    } catch {}
}

# ---------------------------------------------------------------------------
# SECTION 8: Scheduled Tasks
# ---------------------------------------------------------------------------
Write-Section "8. Scheduled Tasks"

$tasks = @(
    @{Path="\Microsoft\Windows\Application Experience\"; Name="Microsoft Compatibility Appraiser"},
    @{Path="\Microsoft\Windows\Application Experience\"; Name="ProgramDataUpdater"},
    @{Path="\Microsoft\Windows\Application Experience\"; Name="StartupAppTask"},
    @{Path="\Microsoft\Windows\Autochk\";                Name="Proxy"},
    @{Path="\Microsoft\Windows\Customer Experience Improvement Program\"; Name="Consolidator"},
    @{Path="\Microsoft\Windows\Customer Experience Improvement Program\"; Name="UsbCeip"},
    @{Path="\Microsoft\Windows\DiskDiagnostic\";         Name="Microsoft-Windows-DiskDiagnosticDataCollector"},
    @{Path="\Microsoft\Windows\Feedback\Siuf\";          Name="DmClient"},
    @{Path="\Microsoft\Windows\Feedback\Siuf\";          Name="DmClientOnScenarioDownload"},
    @{Path="\Microsoft\Windows\Windows Error Reporting\";Name="QueueReporting"},
    @{Path="\Microsoft\Windows\WindowsUpdate\";          Name="Automatic App Update"},
    @{Path="\Microsoft\Windows\Maps\";                   Name="MapsUpdateTask"},
    @{Path="\Microsoft\Windows\Maps\";                   Name="MapsToastTask"},
    @{Path="\Microsoft\Windows\Clip\";                   Name="License Validation"},
    @{Path="\Microsoft\Office\";                         Name="Office ClickToRun Service Monitor"},
    @{Path="\Microsoft\Office\";                         Name="OfficeTelemetryAgentLogOn2016"},
    @{Path="\Microsoft\Office\";                         Name="OfficeTelemetryAgentFallBack2016"}
)

foreach ($task in $tasks) {
    try {
        Disable-ScheduledTask -TaskPath $task.Path -TaskName $task.Name -ErrorAction Stop | Out-Null
        Write-OK "Task disabled: $($task.Name)"
    } catch {
        Write-SKIP "Task not found: $($task.Name)"
    }
}

# ---------------------------------------------------------------------------
# SECTION 9: Explorer and UI Tweaks
# ---------------------------------------------------------------------------
Write-Section "9. Explorer / UI Tweaks"

$explorerAdv = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced"
Set-RegValue $explorerAdv "HideFileExt"           0   # Show file extensions
Set-RegValue $explorerAdv "Hidden"                1   # Show hidden files
Set-RegValue $explorerAdv "ShowTaskViewButton"    0   # Remove Task View button
Set-RegValue $explorerAdv "ShowCopilotButton"     0   # Remove Copilot button
Set-RegValue $explorerAdv "TaskbarAl"             0   # Left-align taskbar (optional)
Set-RegValue $explorerAdv "Start_TrackProgs"      0
Set-RegValue $explorerAdv "Start_TrackEnabled"    0

# Remove widgets
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Dsh" "AllowNewsAndInterests" 0

# Disable Chat (Teams consumer) on taskbar
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\Windows Chat" "ChatIcon" 3

# Disable Search highlights
Set-RegValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\SearchSettings" "IsDynamicSearchBoxEnabled" 0

# Disable lock screen ads
Set-RegValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager" "RotatingLockScreenOverlayEnabled" 0

# Faster menu response
Set-RegValue "HKCU:\Control Panel\Desktop" "MenuShowDelay" "0" String

# Disable error sounds
Set-RegValue "HKCU:\AppEvents\Schemes" "(Default)" ".None" String

# ---------------------------------------------------------------------------
# SECTION 10: Office-Specific Tweaks (runs if Office is installed)
# ---------------------------------------------------------------------------
Write-Section "10. Office Tweaks"

$officeInstalled = Test-Path "HKLM:\SOFTWARE\Microsoft\Office"

if ($officeInstalled) {
    Write-OK "Office detected - applying tweaks"

    # Disable Office telemetry
    Set-RegValue "HKCU:\Software\Microsoft\Office\Common\ClientTelemetry"    "DisableTelemetry" 1
    Set-RegValue "HKCU:\Software\Microsoft\Office\16.0\Common\ClientTelemetry" "DisableTelemetry" 1
    Set-RegValue "HKCU:\Software\Microsoft\Office\16.0\Common\ClientTelemetry" "SendTelemetry"    3

    # Disable Connected Experiences (cloud/AI features)
    Set-RegValue "HKCU:\Software\Policies\Microsoft\Office\16.0\Common\Privacy" "DisconnectedState"            2
    Set-RegValue "HKCU:\Software\Policies\Microsoft\Office\16.0\Common\Privacy" "UserContentDisabled"          2
    Set-RegValue "HKCU:\Software\Policies\Microsoft\Office\16.0\Common\Privacy" "DownloadContentDisabled"      2

    # Disable Office background update service (update manually)
    Disable-Service "ClickToRunSvc"

    # Disable hardware acceleration in Office (smoother on VM virtual GPU)
    Set-RegValue "HKCU:\Software\Microsoft\Office\16.0\Common\Graphics" "DisableHardwareNotification" 1
    Set-RegValue "HKCU:\Software\Microsoft\Office\16.0\Common\Graphics" "DisableAnimations"           1

    # Disable startup screens
    Set-RegValue "HKCU:\Software\Microsoft\Office\16.0\Word\Options"    "DisableBootToOfficeStart" 1
    Set-RegValue "HKCU:\Software\Microsoft\Office\16.0\Excel\Options"   "DisableBootToOfficeStart" 1
    Set-RegValue "HKCU:\Software\Microsoft\Office\16.0\PowerPoint\Options" "DisableBootToOfficeStart" 1

    # Disable LinkedIn integration
    Set-RegValue "HKCU:\Software\Microsoft\Office\16.0\Common"          "LinkedIn_ShowProfileCardInOffice" 0

} else {
    Write-SKIP "Office not yet installed - re-run this section after install, or run Office tweaks manually"
}

# ---------------------------------------------------------------------------
# SECTION 11: Windows Update - Defer and Control
# ---------------------------------------------------------------------------
Write-Section "11. Windows Update"

$wuPolicy = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU"
Set-RegValue $wuPolicy "NoAutoUpdate"          1
Set-RegValue $wuPolicy "AUOptions"             2   # Notify before download
Set-RegValue $wuPolicy "NoAutoRebootWithLoggedOnUsers" 1

# Disable driver updates via Windows Update
Set-RegValue "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate" "ExcludeWUDriversInQualityUpdate" 1

Write-OK "Automatic updates disabled - run Windows Update manually when ready"

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
Write-Section "COMPLETE"

$Duration = (Get-Date) - $StartTime
Write-Host "`n  Done in $([math]::Round($Duration.TotalSeconds))s" -ForegroundColor White
Write-Host "  Log saved to: $LogPath" -ForegroundColor Gray
Write-Host "  Full transcript: $Transcript" -ForegroundColor Gray
Write-Host "`n  !! REBOOT REQUIRED for all changes to take effect !!" -ForegroundColor Yellow
Write-Host ""

Add-Content $LogPath "`nCompleted in $([math]::Round($Duration.TotalSeconds))s at $(Get-Date)"

Stop-Transcript

# Offer reboot
$reboot = Read-Host "`n  Reboot now? (Y/N)"
if ($reboot -eq "Y" -or $reboot -eq "y") {
    Restart-Computer -Force
}
