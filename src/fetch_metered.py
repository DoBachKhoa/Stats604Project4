'''
File to extract data from osf server
Done with AI's help & left as-it.
Extracts the data, unzips and save in the `data/data_metered_raw` directory.
'''
import os
import requests
import tarfile
import zipfile

# -------------------------
# Download from OSF
# -------------------------

url = "https://files.osf.io/v1/resources/Py3u6/providers/osfstorage/?zip="
download_name = "downloaded_file"

print("Downloading...")
r = requests.get(url, stream=True)
r.raise_for_status()

content_type = r.headers.get("Content-Type", "")

# Guess extension from content type
if "gzip" in content_type:
    download_name += ".tar.gz"
elif "tar" in content_type:
    download_name += ".tar"
elif "zip" in content_type:
    download_name += ".zip"
else:
    # OSF often returns tar.gz without correct type header
    download_name += ".tar.gz"

with open(download_name, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

print("Downloaded:", download_name)


# -------------------------
# Recursive extractor
# -------------------------

def extract_nested(path, outdir):
    """
    Extract archive at 'path' into 'outdir'.
    If extraction creates other archives, extract them too.
    Removes intermediate archives.
    """
    # 1. Extract based on type
    if zipfile.is_zipfile(path):
        print(f"Extracting ZIP: {path}")
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(outdir)

    elif tarfile.is_tarfile(path):
        print(f"Extracting TAR: {path}")
        with tarfile.open(path, "r:*") as tar:
            tar.extractall(outdir)

    else:
        print(f"Skipping non-archive: {path}")
        return

    # 2. Remove the archive after extraction
    os.remove(path)

    # 3. Find new archives created and extract them too
    for root, _, files in os.walk(outdir):
        for name in files:
            full = os.path.join(root, name)
            if zipfile.is_zipfile(full) or tarfile.is_tarfile(full):
                extract_nested(full, root)


# -------------------------
# Run extractor
# -------------------------

extract_dir = "data/data_metered_raw"
os.makedirs(extract_dir, exist_ok=True)

extract_nested(download_name, extract_dir)

print("\nAll extraction complete!")
print("Files available in:", extract_dir)
