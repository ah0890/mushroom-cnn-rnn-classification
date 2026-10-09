"""
download_data.py
----------------
Downloads the datasets required for the Mushroom CNN/RNN assignment.

Files:
    data/mushrooms.csv        - Mushroom classification dataset
    data/exercise_text.json   - Text used for the Hugging Face tokenization task

Usage:
    python download_data.py            # download only missing / empty files
    python download_data.py --force    # re-download even if valid files exist

Only the Python standard library is required, so this script can run before
installing the project requirements. If the optional `truststore` package is
installed it is used to verify HTTPS certificates with the OS certificate store.
"""

import argparse
import json
import shutil
import socket
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Folder that holds the data (always relative to this script's location)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# File name -> download URL
DATASETS = {
    "mushrooms.csv": "https://lms.digiskills.pk/Courses/AIP301/Downloads/mushrooms.csv",
    "exercise_text.json": "https://lms.digiskills.pk/Courses/AIP301/Downloads/exercise_text.json",
}

TIMEOUT_SECONDS = 30


def is_valid_file(path: Path) -> bool:
    """Return True if the file exists, is not empty and looks like real data
    (not an HTML error/login page served instead of the dataset)."""
    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return False

    head = text.lstrip()[:200].lower()
    if head.startswith("<!doctype html") or head.startswith("<html"):
        return False

    if path.suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return False

    if path.suffix == ".csv":
        # A CSV should have a header line and at least one data line containing commas
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2 or "," not in lines[0]:
            return False

    return True


def fetch_with_urllib(url: str, context=None) -> bytes:
    """Fetch a URL with urllib. Raises an exception on failure."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS, context=context) as response:
        return response.read()


def fetch_with_os_trust_store(url: str) -> bytes:
    """Fetch a URL while verifying HTTPS certificates with the operating system's
    certificate store (via the optional `truststore` package). The OS can fetch
    missing intermediate certificates, which Python's default check cannot."""
    import truststore  # optional dependency, listed in requirements.txt

    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return fetch_with_urllib(url, context=context)


def fetch_with_curl(url: str) -> bytes:
    """Fetch a URL using the system `curl` command (certificates are still verified)."""
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is not installed")
    result = subprocess.run(
        [curl, "--silent", "--show-error", "--fail", "--location",
         "--max-time", str(TIMEOUT_SECONDS), url],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def fetch(url: str) -> bytes:
    """Download a URL and return its bytes, trying several methods.

    1. Python's standard urllib.
    2. If certificate verification fails (for example because the server does
       not send its intermediate certificate), retry using the OS trust store.
    3. As a last resort, retry with the system `curl` command.
    Certificate verification is never disabled.
    """
    try:
        return fetch_with_urllib(url)
    except urllib.error.HTTPError:
        raise  # a real HTTP error (404, 403, ...) will not be fixed by retrying
    except urllib.error.URLError as err:
        if not isinstance(err.reason, ssl.SSLCertVerificationError):
            raise
        print("  [INFO] Python could not verify the server certificate; "
              "retrying with the operating system certificate store ...")

    try:
        return fetch_with_os_trust_store(url)
    except ImportError:
        print("  [INFO] 'truststore' is not installed; retrying with curl ...")
    except Exception as err:
        print(f"  [INFO] OS trust store attempt failed ({err}); retrying with curl ...")

    return fetch_with_curl(url)


def download_file(url: str, destination: Path) -> bool:
    """Download `url` to `destination`. Returns True on success."""
    temp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        content = fetch(url)
    except urllib.error.HTTPError as err:
        print(f"  [ERROR] HTTP error {err.code} ({err.reason}) for {url}")
        return False
    except urllib.error.URLError as err:
        print(f"  [ERROR] Connection failed for {url}: {err.reason}")
        return False
    except (socket.timeout, TimeoutError):
        print(f"  [ERROR] Timed out after {TIMEOUT_SECONDS}s for {url}")
        return False
    except Exception as err:  # any other unexpected network problem
        print(f"  [ERROR] Download failed for {url}: {err}")
        return False

    if not content:
        print(f"  [ERROR] The server returned an empty file for {url}")
        return False

    # Write to a temporary file first, validate it, then move it into place
    temp_path.write_bytes(content)
    if not is_valid_file(temp_path):
        print(f"  [ERROR] Downloaded content from {url} does not look like a valid "
              f"{destination.suffix} file (it may be an error or login page).")
        temp_path.unlink(missing_ok=True)
        return False

    temp_path.replace(destination)
    print(f"  [OK] Saved {destination.relative_to(PROJECT_ROOT)} ({len(content):,} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download assignment datasets.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download files even if valid copies already exist.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR}\n")

    missing = []
    for filename, url in DATASETS.items():
        destination = DATA_DIR / filename
        print(f"{filename}")

        if is_valid_file(destination) and not args.force:
            print(f"  [SKIP] A valid file already exists "
                  f"({destination.stat().st_size:,} bytes). Use --force to re-download.")
            continue

        print(f"  Downloading from {url} ...")
        if not download_file(url, destination):
            missing.append(filename)

    print()
    if missing:
        print("Some files could not be downloaded automatically.")
        print("Please download them manually and place them in the data/ folder:")
        for filename in missing:
            print(f"  - {DATASETS[filename]}\n      -> data/{filename}")
        return 1

    print("All dataset files are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
