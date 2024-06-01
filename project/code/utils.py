def center_crop(image, target_width, target_height):
    width, height = image.size
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2
    return image.crop((left, top, right, bottom))

def filter_img_path(paths):
    return [x for x in paths if x.split('.')[-1] in ['jpg', 'jpeg']]