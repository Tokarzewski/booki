#!/usr/bin/env bash
# Booki Android dev setup. Idempotent — safe to re-run.
#
# Installs / verifies:
#   - JDK 17  (default on Arch; falls back to apt/brew elsewhere)
#   - Android SDK cmdline-tools + platform-tools + platform-34 + build-tools 34
#   - Writes android/local.properties
#   - Generates Gradle wrapper (Gradle 8.13)
#   - Builds the debug APK
#
# Usage:
#   ./scripts/dev-setup.sh               # full setup + build
#   ./scripts/dev-setup.sh --no-build    # skip the final assembleDebug

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
SDK_VERSION="android-36"
BUILD_TOOLS="36.0.0"
CMDLINE_TOOLS_ZIP="commandlinetools-linux-11076708_latest.zip"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

detect_os() {
    case "$(uname -s)" in
        Linux*)  if [ -f /etc/arch-release ]; then echo arch
                 elif command -v apt-get >/dev/null 2>&1; then echo debian
                 else echo linux; fi ;;
        Darwin*) echo macos ;;
        *)       echo unknown ;;
    esac
}

OS="$(detect_os)"
log "detected OS: $OS"

# ---------------------------------------------------------------------------
# Base tools (curl + unzip)
# ---------------------------------------------------------------------------
ensure_base_tools() {
    local missing=()
    command -v curl  >/dev/null 2>&1 || missing+=(curl)
    command -v unzip >/dev/null 2>&1 || missing+=(unzip)
    [ "${#missing[@]}" -eq 0 ] && return

    log "installing base tools: ${missing[*]}"
    case "$OS" in
        arch)    sudo pacman -S --needed --noconfirm "${missing[@]}" ;;
        debian)  sudo apt-get update && sudo apt-get install -y "${missing[@]}" ;;
        macos)   command -v brew >/dev/null && brew install "${missing[@]}" || die "Install: ${missing[*]}" ;;
        *)       die "Install these manually: ${missing[*]}" ;;
    esac
}
ensure_base_tools

# ---------------------------------------------------------------------------
# JDK 17
# ---------------------------------------------------------------------------
ensure_jdk17() {
    if command -v java >/dev/null 2>&1; then
        ver="$(java -version 2>&1 | awk -F\" '/version/ {print $2}')"
        case "$ver" in 17.*) log "JDK 17 already active ($ver)"; return ;; esac
    fi

    log "installing JDK 17"
    case "$OS" in
        arch)
            sudo pacman -S --needed --noconfirm jdk17-openjdk
            sudo archlinux-java set java-17-openjdk
            ;;
        debian)
            sudo apt-get update
            sudo apt-get install -y openjdk-17-jdk
            sudo update-alternatives --set java "$(update-alternatives --list java | grep -E 'java-17' | head -1)"
            ;;
        macos)
            command -v brew >/dev/null || die "Homebrew not found. Install from https://brew.sh"
            brew install openjdk@17
            sudo ln -sfn "$(brew --prefix openjdk@17)/libexec/openjdk.jdk" /Library/Java/JavaVirtualMachines/openjdk-17.jdk
            ;;
        *)
            die "Unsupported OS '$OS' for automatic JDK install. Install JDK 17 manually."
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Android SDK
# ---------------------------------------------------------------------------
ensure_cmdline_tools() {
    local target="$SDK_HOME/cmdline-tools/latest"
    if [ -x "$target/bin/sdkmanager" ]; then
        log "Android cmdline-tools already present at $target"
        return
    fi
    log "downloading Android cmdline-tools to $target"
    mkdir -p "$SDK_HOME/cmdline-tools"
    local tmp; tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/cmdtools.zip" \
        "https://dl.google.com/android/repository/$CMDLINE_TOOLS_ZIP"
    unzip -q "$tmp/cmdtools.zip" -d "$tmp"
    mv "$tmp/cmdline-tools" "$target"
    rm -rf "$tmp"
}

ensure_sdk_packages() {
    local sdkmanager="$SDK_HOME/cmdline-tools/latest/bin/sdkmanager"
    [ -x "$sdkmanager" ] || die "sdkmanager not found at $sdkmanager"

    log "accepting SDK licenses"
    yes | "$sdkmanager" --licenses >/dev/null || true

    log "installing platforms;$SDK_VERSION + build-tools;$BUILD_TOOLS + platform-tools"
    "$sdkmanager" \
        "platforms;$SDK_VERSION" \
        "build-tools;$BUILD_TOOLS" \
        "platform-tools" >/dev/null
}

write_local_properties() {
    local f="$REPO_ROOT/local.properties"
    if [ -f "$f" ] && grep -q "^sdk.dir=" "$f"; then
        log "local.properties already configured"
    else
        log "writing local.properties (sdk.dir=$SDK_HOME)"
        echo "sdk.dir=$SDK_HOME" > "$f"
    fi
}

persist_env() {
    local rc="${HOME}/.$(basename "${SHELL:-bash}")rc"
    if ! grep -q "ANDROID_HOME" "$rc" 2>/dev/null; then
        log "appending ANDROID_HOME export to $rc"
        {
            echo ""
            echo "# Booki / Android SDK"
            echo "export ANDROID_HOME=\"$SDK_HOME\""
            echo "export PATH=\"\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/platform-tools:\$PATH\""
        } >> "$rc"
    fi
    export ANDROID_HOME="$SDK_HOME"
    export PATH="$SDK_HOME/cmdline-tools/latest/bin:$SDK_HOME/platform-tools:$PATH"
}

# ---------------------------------------------------------------------------
# Gradle wrapper
# ---------------------------------------------------------------------------
ensure_wrapper() {
    cd "$REPO_ROOT"
    if [ ! -f gradle/wrapper/gradle-wrapper.jar ]; then
        log "generating Gradle wrapper"
        if command -v gradle >/dev/null 2>&1; then
            gradle wrapper --gradle-version 8.13
        else
            die "No system 'gradle' to bootstrap the wrapper. Install gradle or run with pre-existing wrapper."
        fi
    else
        log "Gradle wrapper already present"
    fi
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
ensure_jdk17
ensure_cmdline_tools
ensure_sdk_packages
persist_env
write_local_properties
ensure_wrapper

if [ "${1:-}" = "--no-build" ]; then
    log "skipping build (--no-build)"
else
    log "building debug APK"
    cd "$REPO_ROOT"
    ./gradlew :app:assembleDebug
    log "APK: $REPO_ROOT/app/build/outputs/apk/debug/app-debug.apk"
fi

log "done. Connect a device and run: ./gradlew installDebug"
