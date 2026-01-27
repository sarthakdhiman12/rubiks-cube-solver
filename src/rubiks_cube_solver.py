import kociemba

def solve_cube(cube_state_string: str) -> str:
    """
    Solves the Rubik's cube using Kociemba's algorithm.

    Args:
        cube_state_string (str): A 54-character string representing the cube state.
                                 Order: U1...U9, R1...R9, F1...F9, D1...D9, L1...L9, B1...B9

    Returns:
        str: The solution sequence (e.g., "R U R' U'").
    """
    try:
        solution = kociemba.solve(cube_state_string)
        return solution
    except Exception as e:
        raise ValueError(f"Could not solve cube: {str(e)}")
