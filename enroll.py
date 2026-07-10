import argparse
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DATABASE_FILE,
    KNOWN_FACES_DIR,
    MIN_FACE_SIZE,
    create_required_directories,
)
from src.database import FaceDatabase
from src.face_engine import FaceEngine


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a new person in the face database."
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the person to register."
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of face samples to collect."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Camera index."
    )

    return parser.parse_args()


def draw_message(
    frame: np.ndarray,
    message: str,
    position: tuple = (20, 40),
    scale: float = 0.8
) -> None:
    """Draw readable text on an OpenCV frame."""
    cv2.putText(
        frame,
        message,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        message,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def save_face_image(
    frame: np.ndarray,
    box: np.ndarray,
    output_path: Path
) -> None:
    """Crop and save a detected face image."""
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = box.astype(int)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size > 0:
        cv2.imwrite(
            str(output_path),
            face_crop
        )


def main() -> None:
    args = parse_arguments()

    person_name = args.name.strip()

    if not person_name:
        raise ValueError("Person name cannot be empty.")

    if args.samples <= 0:
        raise ValueError(
            "The number of samples must be greater than zero."
        )

    create_required_directories()

    person_directory = KNOWN_FACES_DIR / person_name
    person_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    database = FaceDatabase(DATABASE_FILE)
    database.load()

    engine = FaceEngine(
        min_face_size=MIN_FACE_SIZE
    )

    camera = cv2.VideoCapture(args.camera)

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )
    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not camera.isOpened():
        raise RuntimeError(
            "The camera could not be opened."
        )

    collected_embeddings: List[np.ndarray] = []
    last_capture_time = 0.0
    capture_interval = 0.4

    print()
    print(f"Registering: {person_name}")
    print("Move your head slightly between captures.")
    print("Press Q to cancel.")
    print()

    try:
        while len(collected_embeddings) < args.samples:
            success, frame = camera.read()

            if not success:
                print("Failed to read a camera frame.")
                break

            display_frame = frame.copy()

            (
                boxes,
                embeddings,
                probabilities
            ) = engine.extract_embeddings(frame)

            if len(boxes) == 1:
                box = boxes[0]
                embedding = embeddings[0]
                probability = probabilities[0]

                x1, y1, x2, y2 = box.astype(int)

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                draw_message(
                    display_frame,
                    f"Face confidence: {probability:.2f}",
                    (20, 75),
                    0.7
                )

                current_time = time.time()

                if (
                    probability >= 0.95
                    and current_time - last_capture_time
                    >= capture_interval
                ):
                    normalised_embedding = (
                        engine.normalise_embedding(
                            embedding
                        )
                    )

                    collected_embeddings.append(
                        normalised_embedding
                    )

                    image_number = len(
                        collected_embeddings
                    )

                    image_path = (
                        person_directory
                        / f"{person_name}_{image_number:03d}.jpg"
                    )

                    save_face_image(
                        frame,
                        box,
                        image_path
                    )

                    last_capture_time = current_time

            elif len(boxes) > 1:
                draw_message(
                    display_frame,
                    "Only one person must be visible.",
                    (20, 80),
                    0.8
                )

            else:
                draw_message(
                    display_frame,
                    "No face detected.",
                    (20, 80),
                    0.8
                )

            draw_message(
                display_frame,
                (
                    f"Collected: "
                    f"{len(collected_embeddings)}/{args.samples}"
                ),
                (20, 40),
                0.9
            )

            draw_message(
                display_frame,
                "Press Q to cancel.",
                (20, display_frame.shape[0] - 25),
                0.7
            )

            cv2.imshow(
                "Face Registration",
                display_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Registration cancelled.")
                return

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not collected_embeddings:
        print("No face samples were collected.")
        return

    database.add_embeddings(
        person_name,
        np.array(
            collected_embeddings,
            dtype=np.float32
        )
    )

    database.save()

    print()
    print(
        f"{len(collected_embeddings)} samples "
        f"were registered for {person_name}."
    )
    print(
        f"Database saved to: {DATABASE_FILE}"
    )


if __name__ == "__main__":
    main()