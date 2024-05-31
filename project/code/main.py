import tkinter as tk
import os
from PIL import Image
from PIL import ImageTk as itk
import numpy as np
import itertools
import seaborn as sns
import yaml
from termcolor import cprint

from utlis import center_crop
from poisson import poisson_edit
from mantage import abswap


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

SOURCE_IMG_DIR = config['source_img_dir']
OUTPUT_IMG_DIR = config['output_img_dir']
CANVAS_BORDER_WIDTH = config['canvas_border_width']
BRUSH_WIDTH = config['brush_width']
FIXED_WIDTH = 400
FIXED_HEIGHT = 300

root = tk.Tk()
root.title('Interactive Digital Photomontage')

if not os.path.exists(OUTPUT_IMG_DIR):
    os.mkdir(OUTPUT_IMG_DIR)
def filter_img_path(paths):
    return [x for x in paths if x.split('.')[-1] in ['jpg', 'jpeg']]

SOURCE_IMG_PATHS = filter_img_path(os.listdir(SOURCE_IMG_DIR))
SOURCE_IMG_PATHS = [os.path.join(SOURCE_IMG_DIR, x) for x in SOURCE_IMG_PATHS]
# crop and resize
SOURCE_PIL_IMGS = [center_crop(Image.open(x).convert('RGB'), FIXED_WIDTH, FIXED_HEIGHT) for x in SOURCE_IMG_PATHS]
SOURCE_PHOTOIMAGES = [itk.PhotoImage(image=img) for img in SOURCE_PIL_IMGS]

CURRENT_SOURCE_IDX = 0
COMPOSITE_PATH = os.path.join(OUTPUT_IMG_DIR, 'output.jpg')
COMPOSITE_ARRAY = np.array(SOURCE_PIL_IMGS[0])
Image.fromarray(COMPOSITE_ARRAY).save(COMPOSITE_PATH)
COMPOSITE_PHOTOIMAGE = itk.PhotoImage(file=COMPOSITE_PATH)
LABEL_MAP_PATH = os.path.join(OUTPUT_IMG_DIR, 'label_map.jpg')

hex2rgb = lambda x: tuple(int(x[i:i+2], 16) for i in (1, 3, 5))
rgb2hex = lambda x: '#%02x%02x%02x' % tuple(x)
palette = sns.color_palette()
for i in range(len(palette)):
    color = [int(x*255) for x in palette[i]]
    palette[i] = rgb2hex(color)
palette = itertools.cycle(palette)
COLORS = [next(palette) for i in range(len(SOURCE_PHOTOIMAGES))]


# create composite and source canvas
def canvas_draw(event):
    x, y = event.x, event.y
    w = event.widget
    r = BRUSH_WIDTH
    width, height = w.winfo_width(), w.winfo_height()
    x_min = max(x-r, 0)
    y_min = max(y-r, 0)
    x_max = min(x+r, width-1)
    y_max = min(y+r, height-1)
    color = w.stroke_color
    w.create_rectangle(x_min, y_min, x_max, y_max, fill=color, outline=color)
    w.mask[y_min:y_max+1, x_min:x_max+1] = 1

frame = tk.Frame(root)
frame.pack(padx=5, pady=5, fill=tk.BOTH)
def create_canvas(root):
    canvas = tk.Canvas(root, width=FIXED_WIDTH, height=FIXED_HEIGHT)
    canvas.config(highlightbackground=config['highlight_background'])
    canvas.config(highlightthickness=CANVAS_BORDER_WIDTH)
    canvas.config(background=config['background_color'])

    canvas.pack(padx=5, pady=5, side=tk.LEFT)
    canvas.bind('<B1-Motion>', canvas_draw)
    # set the mask to be an attribute for canvas
    canvas.mask = np.zeros((FIXED_HEIGHT, FIXED_WIDTH), dtype=bool)
    # set the stroke color to be an attribute for canvas
    canvas.stroke_color = 'black'
    return canvas
composite_canvas = create_canvas(frame)
source_canvas = create_canvas(frame)

# create label map canvas
canvas = tk.Canvas(frame, width=FIXED_WIDTH, height=FIXED_HEIGHT)
canvas.config(highlightbackground=config['highlight_background'])
canvas.config(highlightthickness=CANVAS_BORDER_WIDTH)
canvas.config(background=config['background_color'])
canvas.pack(padx=5, pady=5, side=tk.LEFT)
# set the label_map to be an attribute for canvas
canvas.label_map = np.zeros((FIXED_HEIGHT, FIXED_WIDTH), dtype=np.int64)
canvas.label_map_img = None
label_map_canvas = canvas

def update_label_map(label_map, mask, idx):
    '''
    update label map
    '''
    mask = mask.astype(bool)
    label_map[mask] = idx

def show_label_map(canvas):
    '''
    show label map on canvas
    '''
    label_map = np.zeros(shape=[FIXED_HEIGHT, FIXED_WIDTH, 3], dtype=np.uint8)
    for i in range(len(COLORS)):
        color_array = np.array(list(hex2rgb(COLORS[i])))
        for k in range(3):
            m = label_map[:, :, k]
            m[canvas.label_map==i] = color_array[k]
            label_map[:, :, k] = m
    Image.fromarray(label_map).save(LABEL_MAP_PATH)
    canvas.label_map_img = itk.PhotoImage(file=LABEL_MAP_PATH)
    canvas.create_image(CANVAS_BORDER_WIDTH, CANVAS_BORDER_WIDTH, image=canvas.label_map_img, anchor=tk.NW)

# show initial label map
all_true_mask = np.ones_like(label_map_canvas.label_map, dtype=bool)
update_label_map(label_map_canvas.label_map, all_true_mask, 0)
show_label_map(label_map_canvas)

def canvas_show_image(canvas, photoimage):
    '''
    reseting canvas
    '''
    canvas.create_image(CANVAS_BORDER_WIDTH, CANVAS_BORDER_WIDTH, image=photoimage, anchor=tk.NW)

def reset_source():
    canvas = source_canvas
    img_idx = CURRENT_SOURCE_IDX
    canvas_show_image(canvas, SOURCE_PHOTOIMAGES[img_idx])
    canvas.config(highlightbackground=COLORS[img_idx])
    canvas.stroke_color = COLORS[img_idx]
    source_canvas.mask[:, :] = 0

def reset_composite():
    canvas = composite_canvas
    canvas_show_image(canvas, COMPOSITE_PHOTOIMAGE)
    composite_canvas.mask[:, :] = 0

# show initial images on canvas
reset_source()
reset_composite()

# create button
frame = tk.Frame(root)
frame.pack(padx=10, pady=10, fill=tk.BOTH)

# function for creating composite image
def create_composite(binary_map, source_idx):
    binary_map = binary_map.astype(bool)
    global COMPOSITE_ARRAY, COMPOSITE_PATH, COMPOSITE_PHOTOIMAGE
    source = np.array(SOURCE_PIL_IMGS[source_idx])
    target = COMPOSITE_ARRAY
    mask = binary_map.astype(np.uint8) * 255
    COMPOSITE_ARRAY = poisson_edit(source, target, mask, (0, 0))
    # COMPOSITE_ARRAY[binary_map] = np.array(SOURCE_PIL_IMGS[source_idx])[binary_map]
    Image.fromarray(COMPOSITE_ARRAY).save(COMPOSITE_PATH)
    COMPOSITE_PHOTOIMAGE = itk.PhotoImage(file=COMPOSITE_PATH)

# callback for run button
def run_callback():
    composite = COMPOSITE_ARRAY
    source = np.array(SOURCE_PIL_IMGS[CURRENT_SOURCE_IDX])
    binary_map = abswap(composite, source, composite_canvas.mask, source_canvas.mask)
    # update and show label map
    update_label_map(label_map_canvas.label_map, binary_map, CURRENT_SOURCE_IDX)
    show_label_map(label_map_canvas)
    # create and show composite image
    create_composite(binary_map, CURRENT_SOURCE_IDX)
    reset_composite()
    reset_source()

# callback for next button
def next_callback():
    global CURRENT_SOURCE_IDX
    next_idx = (CURRENT_SOURCE_IDX + 1) % len(SOURCE_PHOTOIMAGES)
    CURRENT_SOURCE_IDX = next_idx
    reset_source()

# add widget
run_button = tk.Button(frame, text="Run", command=run_callback)
run_button.pack(padx=10, pady=0, side=tk.LEFT)
next_button = tk.Button(frame, text="Next image", command=next_callback)
next_button.pack(padx=10, pady=0, side=tk.LEFT)
reset_source_button = tk.Button(frame, text="Reset Source", command=reset_source)
reset_source_button.pack(padx=10, pady=0, side=tk.LEFT)
reset_composite_button = tk.Button(frame, text="Reset Composite", command=reset_composite)
reset_composite_button.pack(padx=10, pady=0, side=tk.LEFT)

root.mainloop()
