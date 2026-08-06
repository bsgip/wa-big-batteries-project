import requests
import http
from utils.dirs import data_dir

def download_file(url: str, filename: str):
    res = requests.get(url)
    filepath = data_dir / filename

    if res.status_code == http.HTTPStatus.OK:
        with open(filepath, "wb") as f:
            f.write(res.content)
            print(f'{filename} downloaded successfully')
    else:
        print(f'Failed to download {filename}')



