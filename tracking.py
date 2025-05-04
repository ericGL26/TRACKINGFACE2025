import cv2
import mediapipe as mp

# Inicializando MediaPipe FaceMesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Criando DrawingSpecs customizados
landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0)
connection_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2)

# Captura da webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # para incluir olhos e íris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Falha ao capturar imagem da webcam")
            break

        # Converte BGR para RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Processa a imagem
        results = face_mesh.process(image_rgb)

        # Converte de volta pra BGR pra OpenCV
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Desenha os landmarks se houver
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec
                )

        # Exibe o frame
        cv2.imshow('FaceMesh Tracking', image_bgr)

        # Sai ao apertar ESC (27)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
