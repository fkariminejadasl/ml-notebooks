# Install Ubuntu on a Windows Laptop with NVIDIA

Follow the official installation guide:
https://documentation.ubuntu.com/desktop/en/latest/tutorial/install-ubuntu-desktop

If the laptop does not boot the Ubuntu USB installer, set the BIOS boot priority to USB HDD.

Installation may hard-freeze. Try, in order:

- Normal reboot: Ctrl + Alt + Del
- Safe reboot (REISUB): hold Alt + SysRq (PrtSc), then type slowly: R E I S U B
- Power-off sequence (REISUO): hold Alt + SysRq (PrtSc), then type slowly: R E I S U O
- Last resort: hold the power button for 10 seconds

If freezes keep happening, adjust BIOS settings. Enter BIOS at startup (often Enter then F1, but varies by model; watch the brief on-screen prompt).

- Startup/Boot Order Lock: Off
- Startup/Boot priority: USB HDD
- Security/Secure Boot/Secure Boot: Off
- Security/Secure Boot/Allow Microsoft 3rd party UEFI CA: Off

Install Ubuntu from the USB drive. During install:

- Do not install third-party software
- Do not download updates during install

If the installer freezes or you get a black screen, boot with nomodeset: In GRUB press `e` and add `nomodeset` after `quiet splash` to the Linux line. If you do not see GRUB, hold Shift or tap Esc during startup.

After install, install the latest tested NVIDIA driver (open, not server). Example: `nvidia-driver-590-open (proprietary, tested)`

## Hardware tested example

```bash
Lenovo ThinkPad P16 G3 U9 RTXP3000 64GB/1TB
Display size: 40.6 cm (16.0")
Physical resolution: 1920 x 1200 WUXGA
Processor model: Intel Core Ultra 9 275HX, 2.7 GHz
RAM: 64 GB
Graphics: 12 GB NVIDIA RTX PRO 3000 Blackwell, Intel Graphics
```

## Shortcuts

- GRUB: ofteen hold Shift or tap Esc during startup
- Startup menu: often F12
- BIOS setup: often Enter then F1
- TTY: often Ctrl + Alt + (Fn +) F3, return with Ctrl + Alt + (Fn +) F1. 

## BIOS issue: Intel RST/VMD

The official guide notes that Intel RST (Rapid Storage Technology) is not supported and can block installation.

In BIOS, check Config/Storage. If you see VMD Controller (On/Off) or an Intel Rapid Storage Technology page, your system supports RST/VMD. If VMD/RST is On, turn it Off.

## GRUB change for flickering screen

Temporary edit:

- Tap Esc (or hold Shift) to reach GRUB
- If you land in a grub prompt, type normal and press Enter
- Press e and `add i915.enable_psr=0` after `quiet splash` to the Linux line, then boot

Permanent edit:

```bash
sudo vi /etc/default/grub
sudo update-grub
```

Set:

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash i915.enable_psr=0" # disable Intel PSR to reduce flicker
```

Optional, to always show GRUB:

```bash
GRUB_TIMEOUT_STYLE=menu
GRUB_TIMEOUT=5
```

## Useful commands

```bash
# Secure Boot state
mokutil --sb-state

# GPU mode
sudo prime-select query

# Current kernel boot params
cat /proc/cmdline

# GRUB config
cat /etc/default/grub

# NVIDIA sanity check
nvidia-smi
```

## Note about update-grub warning (os-prober)

If you see: Warning: os-prober will not be executed...
This is normal. It matters only if you want GRUB to auto-detect other OS installs (dual boot).

## Definitions

- GRUB: boot menu and loader that starts Ubuntu and can show OS choices
- BIOS/UEFI: firmware settings for boot order, Secure Boot, storage mode (RST/VMD), etc.
- TTY: text console for recovery when graphics fails
- PSR (Panel Self Refresh): Intel display power-saving feature; disabling can fix flicker


## Post-install setup for a new computer

After Ubuntu is installed and working, set up the essentials for daily use.

- Development: SSH key settings for github and clusters (e.g. fnwi, snellius), git, vscode, miniconda 
- Connection: VPN (Ivanti) 
- Usuful softwares: flameshot, QGIS, dbevear 
- vscode: python, remote-ssh


## Useful troubleshooting commands

Run these commands to collect system information when diagnosing boot, graphics, driver, or Secure Boot issues.


```bash
echo "=== KERNEL ==="
uname -r

echo
echo "=== PCI DEVICES ==="
sudo lspci -nnk | grep -A4 -E 'VGA|3D|Display'

echo
echo "=== NVIDIA / NOUVEAU MODULES ==="
lsmod | grep -E '^nvidia|^nouveau'

echo
echo "=== INTEL MODULES ==="
lsmod | grep -E '^i915|^xe'

echo
echo "=== NVIDIA MODINFO ==="
modinfo nvidia | grep -E '^(filename|version):'

echo
echo "=== DKMS STATUS ==="
dkms status

echo
echo "=== SECURE BOOT ==="
mokutil --sb-state

echo
echo "=== NVIDIA PACKAGES ==="
dpkg -l | grep -E 'nvidia|libnvidia|cuda'

echo
echo "=== OEM PACKAGE ==="
dpkg -l | grep -i oem
apt policy oem-sutton-datu-meta

echo
echo "=== RECENT DMESG NVIDIA/INTEL ==="
sudo dmesg -T | grep -iE 'nvidia|nouveau|i915|xe' | tail -n 200
```