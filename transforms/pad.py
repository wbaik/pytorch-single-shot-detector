from PIL import Image

# Reference:
#     https://github.com/kuangliu/torchcv/tree/master/torchcv/transforms

def pad(img, target_size):
    '''Pad image with zeros to the specified size.

    Args:
      img: (PIL.Image) image to be padded.
      target_size: (tuple) target size of (ow,oh).

    Returns:
      img: (PIL.Image) padded image.

    Reference:
      `tf.image.pad_to_bounding_box`
    '''
    canvas = Image.new('RGB', target_size)
    canvas.paste(img, (0,0))  # paste on the left-up corner
    return canvas
