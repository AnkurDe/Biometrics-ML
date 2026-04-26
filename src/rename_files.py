"""This tool is used to rename files in the files in the data folder. In the format of 1, 2, 3..."""
from pathlib import Path
import re

FOLDER = Path(__file__).parent.parent / "Data"
reg = re.compile(r'\.([^.]+)$')

for folder in FOLDER.iterdir():
    counter = 1
    if folder.is_dir():
        print(folder.name)
        for file in sorted(folder.iterdir()):
            if file.is_file():
                ext = reg.search(file.name)
                new_name = file.with_name(str(counter) + "." + ext.group(1))
                file.rename(new_name)
                counter += 1
