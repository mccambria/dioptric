import matplotlib.font_manager as fm

for font in fm.findSystemFonts():
    if "Roboto" in font:
        print("Found:", font)
