import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
from flask import Flask, jsonify, render_template

app = Flask(__name__)

video_capture = cv2.VideoCapture(0)

# load known faces
atharva_image = face_recognition.load_image_file("faces/atharva.jpeg")
atharva_encoding = face_recognition.face_encodings(atharva_image)[0]

known_face_encodings = [atharva_encoding]
known_face_names = ["Atharva"]

# list of expected student
students = known_face_names.copy()
attendance_log = []

face_locations = []
face_encodings = []

# get current date
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

@app.route("/attendance")
def get_attendance():
    return jsonify(attendance_log)

def recognize_faces():
    global students
    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # recognize faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
                    attendance_log.append({"name": name, "time": current_time})

        cv2.imshow("Attendance Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    f.close()

if __name__ == "__main__":
    from threading import Thread
    Thread(target=recognize_faces, daemon=True).start()
    app.run(debug=True)
