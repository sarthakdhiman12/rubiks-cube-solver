import torch
from torchvision import transforms
from PIL import Image
import kociemba
from model import RubiksFaceModel
import sys

# Map ID to Name for debugging
COLOR_NAMES = {0: 'White', 1: 'Yellow', 2: 'Green', 3: 'Blue', 4: 'Red', 5: 'Orange'}

class RubiksSolver:
    def __init__(self, model_path="model.pth"):
        self.device = torch.device("cpu")
        self.model = RubiksFaceModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict_face(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 2)

        return predicted[0].tolist()

    def solve(self, face_images):
        if len(face_images) != 6:
            raise ValueError("Must provide exactly 6 images.")

        detected_faces = []
        for img_path in face_images:
            detected_faces.append(self.predict_face(img_path))

        face_order = ['U', 'R', 'F', 'D', 'L', 'B']
        color_to_face = {}

        # We need to map detected colors to faces based on CENTERS.
        # detected_faces[i][4] is the center color of face i.
        # But wait, how do we know which face is which in the input?
        # The prompt says "solves the rubukis cube using images of cubes faces".
        # It doesn't strictly say the input order is fixed.
        # BUT, standard convention or UI would typically ask for U, R, F, D, L, B.
        # Or we can identify them by center color if we know which center corresponds to which face in standard orientation.
        # Standard: Up=White, Down=Yellow, Front=Green, Back=Blue, Right=Red, Left=Orange.

        # However, the user might hold the cube differently.
        # Kociemba solver expects the string relative to "Up" and "Front" definition.
        # U1...U9 are stickers on the face that has the Center designated as U.

        # So, the strategy is:
        # 1. Identify the 6 centers. They must be unique and correspond to the 6 colors.
        # 2. Assign U, R, F, D, L, B to the colors based on the input order?
        #    NO. The input order tells us which physical face we are looking at relative to the user's holding.
        #    Actually, usually you scan U, R, F, D, L, B.
        #    So, input[0] IS the 'U' face. Its center defines what color 'U' is.
        #    input[1] IS the 'R' face. Its center defines what color 'R' is.
        #    ...

        for i, face_colors in enumerate(detected_faces):
            center_color_id = face_colors[4]
            # Map the color ID to the Face Name (U, R, F, D, L, B) based on input position
            color_to_face[center_color_id] = face_order[i]

        # Now we construct the string.
        # We iterate through the faces in U, R, F, D, L, B order (which matches input order).
        # For each sticker, we look up its color ID in color_to_face to get the face character.

        cube_string = ""
        for face_colors in detected_faces:
            for color_id in face_colors:
                if color_id not in color_to_face:
                     # This happens if a sticker color was predicted as a color that is NOT one of the 6 centers.
                     # e.g. Model predicts 'Red' but no face has 'Red' center (maybe two faces have 'Orange').
                     # In a perfect world/model, this doesn't happen.
                     # For fallback, we can assign it to the face that has this color as center?
                     # Wait, if color_id is not in color_to_face, it means NO face has this center color.
                     # This implies duplicate centers or missing centers.
                     cube_string += "?"
                else:
                    cube_string += color_to_face[color_id]

        return cube_string, color_to_face

    def run_solver(self, cube_string):
        try:
            solution = kociemba.solve(cube_string)
            return solution
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python rubiks_cube_solver.py img1 img2 img3 img4 img5 img6")
    else:
        solver = RubiksSolver()
        imgs = sys.argv[1:7]
        state, mapping = solver.solve(imgs)
        print(f"Detected State: {state}")
        sol = solver.run_solver(state)
        print(f"Solution: {sol}")
