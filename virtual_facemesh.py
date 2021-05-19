import cv2
import logging
import math
import mediapipe as mp
import numpy as np
import pyvirtualcam

from face_mesh_vertices import FACE_CONNECTIONS_RIGHT_EYEBROW
from face_mesh_vertices import FACE_CONNECTIONS_LEFT_EYEBROW
from face_mesh_vertices import FACE_MESH
from typing import List, Tuple, Union, Dict
from mediapipe.framework.formats import landmark_pb2

width = 1280
height = 720

PRESENCE_THRESHOLD = 0.5
RGB_CHANNELS = 3
VISIBILITY_THRESHOLD = 0.5

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def get_landmarks_points(
        image_rgb: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList):
    if not landmark_list:
        return

    image_rows, image_cols, _ = image_rgb.shape
    idx_to_coordinates: Dict[int, Tuple[int, int]] = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < PRESENCE_THRESHOLD)):
            continue
        landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                      image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    return idx_to_coordinates


def draw_landmarks(
        image_rgb: np.ndarray,
        pts: dict,
        landmark_drawing_spec: mp_drawing.DrawingSpec = mp_drawing.DrawingSpec(color=(252, 196, 15))):
    if image_rgb.shape[2] != RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image_rgb.shape
    for landmark_px in pts.values():
        cv2.circle(image_rgb, landmark_px, landmark_drawing_spec.circle_radius,
                   landmark_drawing_spec.color, landmark_drawing_spec.thickness)
    return


def draw_face_parts(
        img: np.ndarray,
        pts: dict,
        connections: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (0, 0, 0),
        thickness: int = 2):
    points_len = len(pts)
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if not (0 <= start_idx < points_len and 0 <= end_idx < points_len):
            logging.warning("The index are out of the range")
            return
        if start_idx in pts and end_idx in pts:
            cv2.line(img, pts[start_idx], pts[end_idx], color, thickness)


def draw_face_mesh(
        img: np.ndarray,
        pts: dict,
        color: Tuple[int, int, int] = (0, 0, 0),
        thickness: int = 2):
    for poly in FACE_MESH:
        p = poly-1
        np_poly = np.array([[
            pts[p[0, 0]],
            pts[p[1, 0]],
            pts[p[2, 0]]
        ]], np.int32)
        cv2.fillPoly(img, [np_poly], (255, 255, 255))
        cv2.polylines(img, [np_poly], True, (255, 120, 255), 1)
    return


def draw_face_mesh_scheme(
        img: np.ndarray,
        pts: dict):
    if len(pts) != 468:
        return
    for poly in FACE_MESH:
        p = poly-1
        np_poly = np.array([[
            pts[p[0, 0]],
            pts[p[1, 0]],
            pts[p[2, 0]]
        ]], np.int32)
        cv2.polylines(img, [np_poly], True, (254, 0, 253), 1)
    return

def draw_letter(
        img: np.ndarray,
        pts: dict):
    if len(pts) != 468:
        return
    for poly in FACE_MESH:
        p = poly-1
        text = 'Programar es chido'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        color = (72, 254, 22)
        thickness = 1
        line_type = cv2.LINE_AA
        cv2.putText(img, text, (pts[10][0] - 300, pts[10][1] - 50), font, font_scale, color, thickness, line_type)

    return

with pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
    yellow_mikado = (15, 196, 252)
    green_crayola = (116, 181, 90)
    pink_light_carmine = (115, 98, 227)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=green_crayola)
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    mode = 0
    logo = cv2.imread('img/calzada.jpg')
    scale_percent = 10 
    width = int(logo.shape[1] * scale_percent / 100)
    height = int(logo.shape[0] * scale_percent / 100)
    dim = (width, height)  
    logo_resized = cv2.resize(logo, dim, interpolation = cv2.INTER_AREA)
    offset = 10
    
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            points2D = None
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    points2D = get_landmarks_points(
                        image_rgb=image,
                        landmark_list=face_landmarks)
            if points2D:
                if mode == 1:
                    draw_face_mesh_scheme(img=image, pts=points2D)
                elif mode == 2:
                    draw_face_parts(
                        img=image,
                        pts=points2D,
                        connections=FACE_CONNECTIONS_RIGHT_EYEBROW,
                        color=yellow_mikado,
                        thickness=10)
                    draw_face_parts(
                        img=image,
                        pts=points2D,
                        connections=FACE_CONNECTIONS_LEFT_EYEBROW,
                        color=yellow_mikado,
                        thickness=10)
                elif mode == 3:
                    draw_landmarks(
                        image_rgb=image,
                        pts=points2D,
                        landmark_drawing_spec=drawing_spec)
                elif mode == 4:
                    draw_letter(
                        img=image,
                        pts=points2D)
            image[offset:offset + logo_resized.shape[0], offset:offset + logo_resized.shape[1]] = logo_resized 
            cv2.imshow('Facemesh virtual camera', image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cam.send(image)
            key = cv2.waitKey(5)
            if key == 27:
                break
            elif key == 49:
                mode = 0
            elif key == 50:
                mode = 1
            elif key == 51:
                mode = 2
            elif key == 52:
                mode = 3
            elif key == 53:
                mode = 4

    cap.release()
