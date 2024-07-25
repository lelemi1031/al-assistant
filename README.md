# al-assistant


### Install audio tools

```
arch -arm64 /opt/homebrew/bin/brew install portaudio --HEAD  # be sure the install the HEAD otherwise you might get the host error from portaudio for Mac
pip3 install --no-cache-dir --global-option='build_ext' --global-option='-I/opt/homebrew/Cellar/portaudio/HEAD-88ab584/include' --global-option='-L/opt/homebrew/Cellar/portaudio/HEAD-88ab584/lib' pyaudio
```