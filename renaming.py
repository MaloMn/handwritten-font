import os

directory = os.fsencode('glyphs_1')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    os.rename(f'glyphs_1/{filename}', 'glyphs_1/' + filename.replace('_1', ''))
