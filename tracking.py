import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


# Drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Cria opções do modelo
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

# Cria o detector
detector = vision.FaceLandmarker.create_from_options(options)

# Inicia webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Converte para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detecta landmarks
    detection_result = detector.detect(mp_image)

    # Se encontrou rosto
    if detection_result.face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in detection_result.face_landmarks:
            # Converte para o formato NormalizedLandmarkList (pra DrawingUtils)
            landmark_proto = landmark_pb2.NormalizedLandmarkList()
            landmark_proto.landmark.extend([
    landmark_pb2.NormalizedLandmark(
        x=landmark.x,
        y=landmark.y,
        z=landmark.z
    ) for landmark in face_landmarks
])

            # Desenha conexões de malha
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark_proto,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Desenha contornos (boca, olhos, etc)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark_proto,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            # Desenha íris
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark_proto,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

    # Mostra imagem
    cv2.imshow('FaceLandmarker — Full Detail', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
