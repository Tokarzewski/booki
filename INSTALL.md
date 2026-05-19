# Installing Booki via Obtainium

[Obtainium](https://github.com/ImranR98/Obtainium) installs Android apps directly
from their source repos and auto-updates them when new releases ship.

## One-tap install

Open this URL on your Android device with Obtainium installed:

```
obtainium://app/{"id":"dev.booki","url":"https://github.com/Tokarzewski/booki","author":"Tokarzewski","name":"Booki","preferredApkIndex":0}
```

Or manually inside Obtainium:

1. Tap **Add App**
2. Source URL: `https://github.com/Tokarzewski/booki`
3. Source: **GitHub**
4. Tap **Add**

Obtainium will fetch the latest GitHub Release, download `booki-vX.Y.Z.apk`, and
prompt to install. Future releases install with one tap from Obtainium's home
screen.

## Direct APK download

Every tagged release publishes a signed APK on the
[Releases page](https://github.com/Tokarzewski/booki/releases). Grab the
`booki-vX.Y.Z.apk` asset and side-load it.

## Permissions to grant on first run

- **Notifications** — required for the foreground synthesis service.
- **Install unknown apps** — granted to Obtainium (or your browser) when
  side-loading.
- **Media playback** — granted implicitly when the service starts.

## After install: provision the Kokoro model

Booki ships without the TTS model (~330 MB). On first launch the in-app
downloader fetches `kokoro-v1.0.onnx`, `voices.bin`, and `vocab.json` from
Hugging Face into app-private storage. If you'd rather side-load them, see
[`android/README.md`](android/README.md#provisioning-the-kokoro-model).
