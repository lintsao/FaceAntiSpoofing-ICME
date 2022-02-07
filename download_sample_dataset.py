import gdown
from zipfile import ZipFile
import os

url = "https://drive.google.com/u/1/uc?id=1DmVZpIBcYKy87V19poEwviFtmpRJ2ZAH&export=download"
output = "pr_depth_map_256.zip"
gdown.download(url, output)

zf = ZipFile(output, 'r')
zf.extractall("./")
zf.close()

os.remove(output)