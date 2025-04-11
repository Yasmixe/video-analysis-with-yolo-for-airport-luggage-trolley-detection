from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import datetime

app = Flask(__name__)

# Charger le modèle YOLOv8
model = YOLO('best.pt')  # Remplace par ton modèle entraîné
conf_threshold = 0.3

# Stockage des détections de chariots
detected_chariots = []

def gen_frames():
    cap = cv2.VideoCapture(0)  # 0 = webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Prédiction YOLO
        results = model.predict(source=frame, conf=conf_threshold, save=False, verbose=False)
        annotated_frame = results[0].plot()

        # Vérifier les détections
        for box in results[0].boxes.data.tolist():
            cls_id = int(box[5])
            cls_name = model.names[cls_id]

            if cls_name.lower() == "chariot":
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                detected_chariots.append({"timestamp": now})

        # Encoder l'image pour le streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    return jsonify(detected_chariots)

if __name__ == '__main__':
    app.run(debug=True)
