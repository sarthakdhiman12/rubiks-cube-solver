import io
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from .model import ColorClassifier
from .rubiks_cube_solver import solve_cube

app = FastAPI()

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pth"
CLASSES = ["white", "yellow", "red", "orange", "green", "blue"]

# Map colors to Kociemba face notation
# We will determine the mapping dynamically based on center stickers
# Standard orientation:
# U: White, R: Red, F: Green, D: Yellow, L: Orange, B: Blue
# But user might orient differently.
# We need to identify which color corresponds to which face (U, R, F, D, L, B) based on the input order.
# The input order is expected to be: U, R, F, D, L, B faces.
# So the center sticker of image 0 is U color, center of image 1 is R color, etc.

model = ColorClassifier()
if torch.cuda.is_available():
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def extract_patches(image: Image.Image) -> List[Image.Image]:
    """
    Extracts 9 patches from a 3x3 grid face image.
    Assumes the image is cropped to the face boundaries.
    """
    width, height = image.size
    patch_w = width / 3
    patch_h = height / 3

    patches = []
    # Grid order: top-left to bottom-right (row by row)
    for row in range(3):
        for col in range(3):
            left = col * patch_w
            top = row * patch_h
            right = left + patch_w
            bottom = top + patch_h

            # Add a small margin to crop center to avoid borders
            margin_w = patch_w * 0.2
            margin_h = patch_h * 0.2

            patch = image.crop((left + margin_w, top + margin_h, right - margin_w, bottom - margin_h))
            patches.append(patch)
    return patches

def predict_color(patch: Image.Image) -> str:
    img_tensor = transform(patch).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        return CLASSES[predicted.item()]

@app.post("/solve")
async def solve_rubiks_cube(files: List[UploadFile] = File(...)):
    if len(files) != 6:
        raise HTTPException(status_code=400, detail="Exactly 6 images are required (U, R, F, D, L, B).")

    # Order of files: U, R, F, D, L, B
    face_order = ["U", "R", "F", "D", "L", "B"]

    all_colors = [] # List of 54 colors
    face_centers = {} # Map color -> Face Notation (U, R, F, D, L, B)

    # Process each face
    for idx, file in enumerate(files):
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        patches = extract_patches(image)
        if len(patches) != 9:
             raise HTTPException(status_code=500, detail=f"Failed to extract 9 patches from image {idx}")

        face_colors = []
        for p_idx, patch in enumerate(patches):
            color = predict_color(patch)
            face_colors.append(color)

            # The 5th patch (index 4) is the center
            if p_idx == 4:
                # Map the detected center color to the current face notation
                # E.g. if image 0 (U) center is 'white', then 'white' maps to 'U'
                if color in face_centers and face_centers[color] != face_order[idx]:
                    # Duplicate center colors detected? This is invalid for a standard cube
                    # But we'll proceed and let Kociemba validation handle it or just overwrite
                    pass
                face_centers[color] = face_order[idx]

        all_colors.extend(face_colors)

    # Validate that we found 6 unique center colors
    if len(face_centers) != 6:
        # Fallback for standard coloring if detection failed or lighting is bad
        # This is risky, but let's try to infer missing ones or error out
        # For now, let's construct the string. If a color is missing from map, we can't solve.
        detected_centers = list(face_centers.keys())
        missing = [c for c in CLASSES if c not in detected_centers]
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not detect all 6 unique center colors. Detected: {detected_centers}"}
        )

    # Construct state string
    # Kociemba expects: UUUUUUUUU RRRRRRRRR FFFFFFFFF DDDDDDDDD LLLLLLLLL BBBBBBBBB
    # Our all_colors is in that order (0-8 is U, 9-17 is R, etc.)
    # We map each color name to the face notation

    state_string = ""
    for color in all_colors:
        if color not in face_centers:
             return JSONResponse(status_code=400, content={"error": f"Detected color {color} which is not a center color."})
        state_string += face_centers[color]

    try:
        solution = solve_cube(state_string)
        return {"solution": solution, "state": state_string}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e), "state": state_string})

@app.get("/")
def read_root():
    return {"message": "Rubik's Cube Solver API. Go to /static/index.html to use the interface."}
