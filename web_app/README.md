# SynthLang-Website
Commands to run the website:

# on windows:

# Make sure WSL is current, then reset it
wsl --update
wsl --shutdown

# sync uv
uv sync
# 1. Start the VM (skip if `podman machine list` shows it running)
podman machine init
podman machine start

# 2. Build the image (only needed when pyproject.toml/uv.lock/Containerfile change)
podman build -t synthlang -f Containerfile .

# 3. Run the app
podman run --rm -it -p 8000:8000 -v "${PWD}:/app" synthlang

# on linux
sudo apt install podman        # or: sudo dnf install podman
cd /web_app
podman build -t synthlang -f Containerfile .
podman run -d --name synthlang \
  --restart=always \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD":/app:Z \
  synthlang