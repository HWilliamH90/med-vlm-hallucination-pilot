"""Step 0: generate a synthetic test image with known content.

The image contains exactly one red circle and one blue rectangle on a white
background, which gives us a known ground truth for the first sanity check.

Run:
    python scripts/00_make_test_image.py
"""
from PIL import Image, ImageDraw

img = Image.new("RGB", (640, 480), "white")
draw = ImageDraw.Draw(img)

# One red circle (ellipse with equal axes).
draw.ellipse((80, 120, 280, 320), fill="red")

# One blue rectangle.
draw.rectangle((360, 140, 560, 340), fill="blue")

img.save("images/test.jpg", quality=95)
print("Saved images/test.jpg with one red circle and one blue rectangle.")
