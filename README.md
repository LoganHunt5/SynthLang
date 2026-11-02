# SynthLang-Website
Instructions for website
Create a language directory in the models folder
    models go in the directory of the language they belong to
each model should have a json file called config.json and any .pth .pt or .pth.tar file


Commands to run the website:

# on windows:

## Make sure WSL is current, then reset it
wsl --update
wsl --shutdown

## sync uv
uv sync
## 1. Start the VM (skip if `podman machine list` shows it running)
podman machine init
podman machine start

## 2. Build the image (only needed when pyproject.toml, uv.lock, or Containerfile change)
podman build -t synthlang -f Containerfile .

## 3. Run the app
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


## note
    removing models while website is running does not free up ram