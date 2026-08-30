"""Generates synthetic receipt images to validate the pipeline end-to-end,
since the actual Google Drive dataset isn't reachable in this sandbox
(no network access). Not part of the deliverable -- just a test fixture."""

import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "/home/claude/carbon_crunch_ocr/sample_data"
os.makedirs(OUT_DIR, exist_ok=True)

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

RECEIPTS = [
    {
        "store": "GREEN LEAF GROCERY",
        "date": "14/03/2024",
        "items": [("Organic Bananas", 2.49), ("Whole Milk 1L", 1.89),
                   ("Brown Bread", 3.20), ("Free Range Eggs", 4.10)],
        "tax": 0.95,
    },
    {
        "store": "URBAN COFFEE CO.",
        "date": "2024-05-02",
        "items": [("Cappuccino Large", 4.50), ("Blueberry Muffin", 3.00),
                   ("Espresso Shot", 2.25)],
        "tax": 0.68,
    },
    {
        "store": "QUICKMART SUPERSTORE",
        "date": "22 Jun 2024",
        "items": [("AA Batteries 4pk", 6.99), ("Notebook A5", 2.49),
                   ("USB Cable 1m", 5.99), ("Hand Sanitizer", 3.49),
                   ("Printer Paper", 7.99)],
        "tax": 1.84,
    },
    {
        # Deliberately sparse / partially-illegible receipt (edge case)
        "store": "CORNER DINER",
        "date": "9/1/24",
        "items": [("Coffee", 1.50)],
        "tax": 0.10,
    },
]


def render_receipt(spec: dict) -> Image.Image:
    width = 420
    lines = []
    font_title = ImageFont.truetype(FONT_BOLD, 22)
    font_body = ImageFont.truetype(FONT_REG, 16)

    subtotal = sum(p for _, p in spec["items"])
    total = round(subtotal + spec["tax"], 2)

    height = 160 + 26 * len(spec["items"]) + 140
    img = Image.new("L", (width, height), color=250)
    draw = ImageDraw.Draw(img)

    y = 20
    draw.text((width / 2, y), spec["store"], font=font_title, fill=10, anchor="ma")
    y += 34
    draw.text((20, y), f"Date: {spec['date']}", font=font_body, fill=20)
    y += 22
    draw.text((20, y), "-" * 40, font=font_body, fill=40)
    y += 24

    for name, price in spec["items"]:
        line = f"{name:<26}{price:>6.2f}"
        draw.text((20, y), line, font=font_body, fill=15)
        y += 26

    draw.text((20, y), "-" * 40, font=font_body, fill=40)
    y += 22
    draw.text((20, y), f"{'Subtotal':<26}{subtotal:>6.2f}", font=font_body, fill=15)
    y += 22
    draw.text((20, y), f"{'Tax':<26}{spec['tax']:>6.2f}", font=font_body, fill=15)
    y += 22
    draw.text((20, y), f"{'TOTAL':<26}{total:>6.2f}", font=font_body, fill=0)
    y += 30
    draw.text((width / 2, y), "THANK YOU!", font=font_body, fill=30, anchor="ma")

    return img


def add_realworld_noise(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    arr = np.array(img).astype(np.float32)

    # Gaussian sensor noise
    noise = np.random.default_rng(seed).normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    # Uneven lighting: add a soft gradient
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = 25 * np.sin(xx / w * np.pi) * rng.choice([-1, 1])
    arr = np.clip(arr + gradient, 0, 255)

    img2 = Image.fromarray(arr.astype(np.uint8))

    # Slight blur (mimics camera shake / focus issues)
    from PIL import ImageFilter
    img2 = img2.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.0)))

    # Random small rotation (skew)
    angle = rng.uniform(-6, 6)
    img2 = img2.rotate(angle, expand=True, fillcolor=250)

    return img2


if __name__ == "__main__":
    for i, spec in enumerate(RECEIPTS, start=1):
        clean = render_receipt(spec)
        noisy = add_realworld_noise(clean, seed=i)
        path = os.path.join(OUT_DIR, f"receipt_{i:02d}.png")
        noisy.convert("L").save(path)
        print("wrote", path, noisy.size)
