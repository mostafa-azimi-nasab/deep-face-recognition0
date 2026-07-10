import time
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DATABASE_FILE,
    FRAME_SKIP,
    MIN_FACE_SIZE,
    RECOGNITION_THRESHOLD,
    create_required_directories,
)
from src.database import FaceDatabase
from src.face_engine import FaceEngine


def draw_label(
    frame: np.ndarray,
    box: np.ndarray,
    label: str,
    distance: float
) -> None:
    """Draw a face box and recognition label."""
    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = box.astype(int)

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))

    if label == "Unknown":
        box_colour = (0, 0, 255)
    else:
        box_colour = (0, 255, 0)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_colour,
        2
    )

    if np.isfinite(distance):
        text = f"{label} | distance: {distance:.2f}"
    else:
        text = label

    text_size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    text_width, text_height = text_size

    label_top = max(
        0,
        y1 - text_height - 15
    )

    cv2.rectangle(
        frame,
        (x1, label_top),
        (
            min(frame_width - 1, x1 + text_width + 10),
            y1
        ),
        box_colour,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x1 + 5, y1 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def recognise_face(
    database: FaceDatabase,
    engine: FaceEngine,
    embedding: np.ndarray
) -> Tuple[str, float]:
    """Compare an embedding with the saved database."""
    normalised_embedding = engine.normalise_embedding(
        embedding
    )

    name, distance = database.find_nearest(
        normalised_embedding
    )

    if name is None:
        return "Unknown", distance

    if distance <= RECOGNITION_THRESHOLD:
        return name, distance

    return "Unknown", distance


def display_status(
    frame: np.ndarray,
    fps: float,
    database_size: int
) -> None:
    """Display FPS and keyboard instructions."""
    status_text = (
        f"FPS: {fps:.1f} | "
        f"Saved samples: {database_size}"
    )

    cv2.putText(
        frame,
        status_text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Q: quit | R: reload database",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def main() -> None:
    create_required_directories()

    database = FaceDatabase(DATABASE_FILE)
    database.load()

    if database.is_empty():
        print(
            "The face database is empty."
        )
        print(
            "Register a person first, for example:"
        )
        print(
            'python enroll.py --name "Behzad" --samples 20'
        )
        return

    engine = FaceEngine(
        min_face_size=MIN_FACE_SIZE
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)

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

    frame_number = 0
    recent_times = deque(maxlen=30)

    previous_results: Dict[
        int,
        Tuple[np.ndarray, str, float]
    ] = {}

    print()
    print("Face recognition is running.")
    print("Press Q to quit.")
    print("Press R to reload the database.")
    print()

    try:
        while True:
            start_time = time.time()

            success, frame = camera.read()

            if not success:
                print("Failed to read a camera frame.")
                break

            frame_number += 1

            should_process = (
                frame_number % (FRAME_SKIP + 1) == 0
                or not previous_results
            )

            if should_process:
                (
                    boxes,
                    embeddings,
                    probabilities
                ) = engine.extract_embeddings(frame)

                previous_results = {}

                for index, (
                    box,
                    embedding,
                    probability
                ) in enumerate(
                    zip(
                        boxes,
                        embeddings,
                        probabilities
                    )
                ):
                    if probability < 0.90:
                        continue

                    label, distance = recognise_face(
                        database,
                        engine,
                        embedding
                    )

                    previous_results[index] = (
                        box,
                        label,
                        distance
                    )

            for box, label, distance in (
                previous_results.values()
            ):
                draw_label(
                    frame,
                    box,
                    label,
                    distance
                )

            elapsed_time = time.time() - start_time
            recent_times.append(elapsed_time)

            average_time = sum(recent_times) / len(
                recent_times
            )

            fps = (
                1.0 / average_time
                if average_time > 0
                else 0.0
            )

            display_status(
                frame,
                fps,
                len(database.names)
            )

            cv2.imshow(
                "Deep Face Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                database.load()
                print(
                    "The face database was reloaded."
                )

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()