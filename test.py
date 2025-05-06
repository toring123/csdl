import cv2
import dlib

# Đường dẫn tới file dự đoán landmark và ảnh khuôn mặt
predictor_path = 'shape_predictor_68_face_landmarks.dat'
image_path = 'anh1\\2.jpg'

# Khởi tạo detector và predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Đọc ảnh và chuyển sang grayscale
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Phát hiện khuôn mặt
faces = detector(gray)

for face in faces:
    landmarks = predictor(gray, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        # Vẽ chấm tròn tại mỗi điểm landmark
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        # Ghi số thứ tự của điểm landmark
        cv2.putText(img, str(n), (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

cv2.imshow("Facial Landmarks with Index", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
