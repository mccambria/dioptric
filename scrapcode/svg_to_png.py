import cairosvg
import os
from os import listdir
from os.path import isfile

def get_file_name(file_path):
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)
    file_name = file_name[0]
    return file_name

def main(src, dest):
    
    for f in listdir(src):
        svg_path = src + f
        if not isfile(svg_path):
            continue
        file_name = get_file_name(f)
        cairosvg.svg2png(url=svg_path, write_to=dest + "{}.png".format(file_name))


if __name__ == "__main__":
    
    src = "/home/mccambria/lab/high_temp_oscillations/svgs/"
    dest = "/home/mccambria/lab/high_temp_oscillations/pngs/"
    
    main(src, dest)
    