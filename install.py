import os
import platform
import shutil
import subprocess


def _run(cmd):
    try:
        return subprocess.run(cmd, check=False)
    except Exception as e:
        print("Command failed:", e)
        return None


def main():
    if shutil.which("exiftool"):
        print("exiftool already available in PATH")
        return

    if os.environ.get("EXIFTOOL_AUTO_INSTALL", "1").lower() in {"0", "false", "no"}:
        print("EXIFTOOL_AUTO_INSTALL disabled. Please install exiftool manually.")
        return

    system = platform.system().lower()

    if system == "linux":
        print("Attempting to install exiftool via apt-get...")
        _run(["apt-get", "update", "-y"])
        _run(["apt-get", "install", "-y", "libimage-exiftool-perl"])
        if shutil.which("exiftool"):
            print("exiftool installed successfully")
        else:
            print("exiftool install failed. Please install manually.")
        return

    if system == "darwin":
        print("Please install exiftool with: brew install exiftool")
        return

    if system == "windows":
        print("Please install exiftool and add it to PATH.")
        print("Example (Chocolatey): choco install exiftool")
        return

    print("Unsupported OS. Please install exiftool manually.")


if __name__ == "__main__":
    main()
