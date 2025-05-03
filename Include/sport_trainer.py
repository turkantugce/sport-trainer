import cv2
import mediapipe as mp
import numpy as np
import math

# PoseEstimation sınıfı, vücut pozisyonlarını algılamak ve izlemek için oluşturuldu.
class PoseEstimation:
    def __init__(self, static_image_mode=False, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # MediaPipe Pose ve çizim yardımcılarını başlat.
        self.static_image_mode = static_image_mode
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose # Pose çözümünü başlat.
        self.mpDraw = mp.solutions.drawing_utils # Çizim yardımcılarını kullan.
        self.pose = self.mpPose.Pose(
            static_image_mode=self.static_image_mode,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    # Pozu algıla ve isteğe bağlı olarak vücut hatlarını çiz.
    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB'ye dönüştür, MediaPipe RGB bekler.
        self.results = self.pose.process(imgRGB)  # Poz analizi yap.

        if self.results.pose_landmarks:  # Poz bulunduysa
            if draw:
                # Vücut bağlantılarını çiz.
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    # Vücut pozisyonundaki eklem noktalarını bul.
    def find_position(self, img, draw=True):
        self.landmark_list = [] # Eklem noktalarını saklamak için liste.
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, _ = img.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                self.landmark_list.append([id, x, y])
                if draw and id == 0:  # Draw on the nose for debugging
                    cv2.circle(img, (x, y), 10, (255, 0, 0), cv2.FILLED)
        return img, self.landmark_list

    # Üç eklem noktası arasında açıyı hesapla.
    def find_angle(self, img, lm1, lm2, lm3, draw=True):
        # Get the landmarks
        x1, y1 = self.landmark_list[lm1][1:]
        x2, y2 = self.landmark_list[lm2][1:]
        x3, y3 = self.landmark_list[lm3][1:]

        # Açı hesaplama (trigonometri).
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            # Noktalar ve çizgiler ile açıyı görselleştir.
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 + 20, y2 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img, angle

# Egzersiz mantığı: açıya göre tekrar sayımı ve görsel geri bildirim.
def exercise_logic(img, angle, exercise_name, range_angle, count, direction):
    min_angle, max_angle = range_angle

    percentage = np.interp(angle, (min_angle, max_angle), (0, 100))
    color = (255, 0, 255)

    if percentage == 100:
        color = (0, 255, 0)
        if direction == 0:
            count += 0.5
            direction = 1

    if percentage == 0:
        color = (0, 255, 0)
        if direction == 1:
            count += 0.5
            direction = 0

    # Bar ve yüzdeyi göster.
    bar = np.interp(angle, (min_angle, max_angle), (450, 200))
    cv2.rectangle(img, (580, int(bar)), (600, 450), color, cv2.FILLED)
    cv2.putText(img, f"{int(percentage)}%", (540, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Tekrar sayısını göster.
    cv2.putText(img, f"{exercise_name}: {int(count)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    return img, count, direction


def main():
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("Pose Tracker", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Pose Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    pose_estimator = PoseEstimation()
    exercise = "arm_curl_right"  # Default exercise
    counts = {"arm_curl_right": 0, "arm_curl_left": 0, "squat": 0, "pushup": 0, "situp": 0}
    directions = {"arm_curl_right": 0, "arm_curl_left": 0, "squat": 0, "pushup": 0, "situp": 0}

    while True:
        ret, img = video_capture.read()
        if not ret:
            break

        img = cv2.resize(img, (640, 480))
        img = pose_estimator.find_pose(img, draw=True)
        img, landmark_list = pose_estimator.find_position(img, draw=False)

        if landmark_list:
            if exercise == "arm_curl_right":
                # Sağ kol için açı
                img, angle_right = pose_estimator.find_angle(img, 12, 14, 16)
                img, counts["arm_curl_right"], directions["arm_curl_right"] = exercise_logic(
                    img, angle_right, "Arm Curl (Right)", (190, 270),
                    counts["arm_curl_right"], directions["arm_curl_right"]
                )

            elif exercise == "arm_curl_left":
                # Sol kol için açı
                img, angle_left = pose_estimator.find_angle(img, 11, 13, 15)
                img, counts["arm_curl_left"], directions["arm_curl_left"] = exercise_logic(
                    img, angle_left, "Arm Curl (Left)", (170, 90),
                    counts["arm_curl_left"], directions["arm_curl_left"]
                )

            elif exercise == "squat":
                img, angle = pose_estimator.find_angle(img, 24, 26, 28)
                img, counts["squat"], directions["squat"] = exercise_logic(
                    img, angle, "Squat", (90, 170),
                    counts["squat"], directions["squat"]
                )

            elif exercise == "pushup":
                img, angle = pose_estimator.find_angle(img, 12, 14, 16)
                img, counts["pushup"], directions["pushup"] = exercise_logic(
                    img, angle, "Pushup", (70, 160),
                    counts["pushup"], directions["pushup"]
                )

            elif exercise == "situp":
                img, angle = pose_estimator.find_angle(img, 11, 23, 25)
                img, counts["situp"], directions["situp"] = exercise_logic(
                    img, angle, "Situp", (120, 200),
                    counts["situp"], directions["situp"]
                )

        # Gösterilen egzersiz ismi
        cv2.putText(img, f"Exercise: {exercise.replace('_', ' ').capitalize()}",
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Ekran görüntüsü
        cv2.imshow("Pose Tracker", img)

        # Klavye Girişleri
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Çıkış
            break
        elif key == ord('1'):  # Sağ Kol Arm Curl
            exercise = "arm_curl_right"
        elif key == ord('2'):  # Sol Kol Arm Curl
            exercise = "arm_curl_left"
        elif key == ord('3'):  # Squat
            exercise = "squat"
        elif key == ord('4'):  # Pushup
            exercise = "pushup"
        elif key == ord('5'):  # Situp
            exercise = "situp"

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
