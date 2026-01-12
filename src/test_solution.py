import os
import random
import sys
import numpy as np
from PIL import Image, ImageDraw

COLORS_RGB = {
    0: (255, 255, 255), # White
    1: (255, 255, 0),   # Yellow
    2: (0, 155, 72),    # Green
    3: (0, 69, 173),    # Blue
    4: (185, 0, 0),     # Red
    5: (255, 89, 0)     # Orange
}

SOLVED_STATE = {
    'U': [0]*9,
    'R': [4]*9,
    'F': [2]*9,
    'D': [1]*9,
    'L': [5]*9,
    'B': [3]*9
}

def generate_face_image(color_ids, filename):
    img_size = 128
    cell_size = img_size // 3
    img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for i, cid in enumerate(color_ids):
        r = i // 3
        c = i % 3
        rgb = COLORS_RGB[cid]
        jitter = lambda x: max(0, min(255, x + random.randint(-5, 5)))
        rgb_j = (jitter(rgb[0]), jitter(rgb[1]), jitter(rgb[2]))

        x0 = c * cell_size
        y0 = r * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size

        draw.rectangle([x0, y0, x1, y1], fill=rgb_j, outline=(0,0,0), width=2)
        gap = 2
        draw.rectangle([x0+gap, y0+gap, x1-gap, y1-gap], fill=rgb_j)

    img.save(filename)

def test_solved_cube():
    print("Generating images for a SOLVED cube...")
    faces = ['U', 'R', 'F', 'D', 'L', 'B']
    filenames = []

    if not os.path.exists("test_data"):
        os.makedirs("test_data")

    for f in faces:
        fname = f"test_data/{f}.png"
        generate_face_image(SOLVED_STATE[f], fname)
        filenames.append(fname)

    print("Running solver...")
    from rubiks_cube_solver import RubiksSolver
    solver = RubiksSolver()

    state_str, mapping = solver.solve(filenames)
    print(f"Detected State String: {state_str}")

    solution = solver.run_solver(state_str)
    print(f"Solution: '{solution}'")

    if not solution or len(solution.strip()) == 0:
        print("SUCCESS: Cube identified as solved.")
    else:
        expected = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
        if state_str == expected:
             print("SUCCESS: State string matches solved state perfectly.")
        else:
             print("FAILURE: State string does not match solved state.")
             print(f"Expected: {expected}")
             print(f"Got:      {state_str}")

if __name__ == "__main__":
    test_solved_cube()
