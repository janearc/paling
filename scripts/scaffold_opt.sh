#!/bin/bash
set -e

echo "=== Paling Deployment Scaffolder ==="

# 1. Create the dedicated OS user
if id "_paling" &>/dev/null; then
    echo "[✓] User '_paling' already exists."
else
    echo "[+] Creating dedicated OS user '_paling'..."
    # macOS system/daemon accounts conventionally start with an underscore
    sudo sysadminctl -addUser _paling -fullName "Paling Orchestration Service" -home /var/empty -shell /usr/bin/false
fi

# 2. Scaffold the /opt/paling hierarchy
echo "[+] Scaffolding /opt/paling directory structure..."
sudo mkdir -p /opt/paling/bin
sudo mkdir -p /opt/paling/lib
sudo mkdir -p /opt/paling/var/bentos
sudo mkdir -p /opt/paling/var/logs
sudo mkdir -p /opt/paling/etc

# 3. Secure the permissions
echo "[+] Setting ownership and permissions..."
# The service user owns everything in /opt/paling
sudo chown -R _paling:staff /opt/paling

# The drop-zone (var/bentos) must be writable by the staff group so developers can submit jobs
sudo chmod -R 775 /opt/paling/var/bentos
sudo chmod -R 775 /opt/paling/var/logs

echo "[✓] Directory scaffolding complete."
echo "
Deployment Structure:
/opt/paling/
├── bin/          (Executables & adapters)
├── lib/          (Isolated Python environment / uv venv)
├── var/
│   ├── bentos/   (The Hadoop-style drop zone for datasets)
│   └── logs/     (Daemon telemetry)
└── etc/          (Configuration)
"
