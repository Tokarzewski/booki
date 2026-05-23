# Booki

Mobile app for ebooks and AI generated audiobooks based on [audiblez](https://github.com/santinic/audiblez).  

## Install

The easiest way is [Obtainium](https://github.com/ImranR98/Obtainium) — add
`https://github.com/Tokarzewski/booki` as a GitHub source and it auto-updates
whenever a new release ships. See [`INSTALL.md`](INSTALL.md) for the one-tap URL.

For side-loading, grab the APK from the
[Releases page](https://github.com/Tokarzewski/booki/releases).

## Development

```bash
cd android
./scripts/dev-setup.sh        # one-shot: JDK 17 + Android SDK + build
./gradlew :app:assembleDebug
```

See [`android/README.md`](android/README.md) for architecture, Kokoro model
provisioning, and known gaps.

## Releasing

1. Run `android/scripts/release-keystore.sh` once to mint a signing key.
2. Add the four secrets it prints to the GitHub repo
   (`KEYSTORE_BASE64`, `KEYSTORE_PASSWORD`, `KEY_ALIAS`, `KEY_PASSWORD`).
3. Tag and push:
   ```bash
   git tag v0.1.0 && git push --tags
   ```
4. GitHub Actions builds + signs the APK and publishes it as a GitHub Release;
   Obtainium picks it up automatically.
