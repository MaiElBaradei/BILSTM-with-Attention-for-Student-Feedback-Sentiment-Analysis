import os
import requests
from tqdm import tqdm

def download_data(
    url: str = "https://data.mendeley.com/public-files/datasets/fvtfjyvw7d/files/256a4429-4fc3-4872-9a7c-26b44a820a8c/file_downloaded",
):
    os.makedirs("data", exist_ok=True)
    destination = os.path.join("data", "data.csv")
    if os.path.exists(destination):
        print("[*] Dataset already downloaded, skipping.")
        return

    print("[*] Downloading dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))

        with (
            open(destination, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading...",
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))