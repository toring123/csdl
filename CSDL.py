import mysql.connector
import numpy as np
from scipy.spatial import KDTree
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import dlib

# 🔹 Khởi tạo detector & predictor của dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Đường dẫn đến file model

# 🔹 Hàm trích xuất đặc trưng từ ảnh
def extract_facial_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        return None

    for face in faces:
        landmarks = predictor(gray, face)

        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

        # Tính toán các đặc trưng
        features = [
            np.linalg.norm(points[42] - points[39]),   # Khoảng cách hai mắt
            np.linalg.norm(points[17] - points[21]),   # Chiều dài cung mày trái
            np.linalg.norm(points[22] - points[26]),   # Chiều dài cung mày phải
            np.linalg.norm(points[21] - points[22]),   # Khoảng cách giữa hai lông mày
            np.linalg.norm(points[48] - points[54]),   # Chiều rộng miệng
            np.linalg.norm(points[51] - points[57]),   # Chiều cao miệng
            np.linalg.norm(points[37] - points[41]),   # Độ mở mắt trái
            np.linalg.norm(points[43] - points[47]),   # Độ mở mắt phải
            np.linalg.norm(points[27] - points[33]),   # Chiều dài sống mũi
            np.linalg.norm(points[31] - points[35]),   # Chiều rộng mũi
            np.linalg.norm(points[0] - points[16]) / np.linalg.norm(points[8] - points[27]),  # Tỉ lệ mặt
            np.arctan2(points[8][1] - points[0][1], points[8][0] - points[0][0])  # Độ dốc hàm
        ]
        return np.array(features)

# 🔹 Hàm tải dữ liệu từ MySQL
def load_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="face_img_db"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT path, eye_distance, left_eyebrow_length, right_eyebrow_length, brow_distance, "
                   "mouth_width, mouth_height, left_eye_opening, right_eye_opening, "
                   "nose_length, nose_width, face_ratio, jaw_slope FROM face_features")
    
    rows = cursor.fetchall()
    paths = [row[0] for row in rows]
    data = np.array([row[1:] for row in rows])

    cursor.close()
    conn.close()
    return paths, data

# 🔹 Hàm tìm ảnh gần nhất
def find_nearest_image(query_point, k):
    distances, indices  = kd_tree.query([query_point], k = k)

    nearest_paths = [paths[i] for i in indices[0]]

    # Hiển thị ảnh gần nhất
    for i in range(0,3):
        img = Image.open(nearest_paths[i]).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        result_image_labels[i].config(image=img)
        result_image_labels[i].image = img
        result_labels[i].config(text=f"Ảnh gần thứ {i+1}: {nearest_paths[i]}\nKhoảng cách: {distances[0][i]:.2f}")

# 🔹 Hàm chọn ảnh từ máy tính
def choose_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    # Hiển thị ảnh đầu vào
    img = Image.open(file_path).resize((200, 200))
    img = ImageTk.PhotoImage(img)
    input_image_label.config(image=img)
    input_image_label.image = img

    # Trích xuất đặc trưng & tìm ảnh gần nhất
    query_features = extract_facial_features(file_path)
    if query_features is not None:
        find_nearest_image(query_features, 3)
    else:
        messagebox.showerror("Lỗi", "Không phát hiện khuôn mặt trong ảnh!")

# 🔹 Tải dữ liệu MySQL và tạo KD-tree
paths, data = load_data()
kd_tree = KDTree(data)

# 🔹 Giao diện Tkinter
root = tk.Tk()
root.title("Tìm kiếm ảnh bằng KD-tree")

btn_choose = tk.Button(root, text="Chọn ảnh", command=choose_image)
btn_choose.pack(pady=10)

input_image_label = tk.Label(root)
input_image_label.pack(pady=5)

result_labels = [tk.Label(root, text="", fg="blue"),tk.Label(root, text="", fg="blue"),tk.Label(root, text="", fg="blue")]

result_image_labels = [tk.Label(root),tk.Label(root),tk.Label(root)]
for i in range(0,3):
    result_labels[i].pack(pady=i*6+5)
    result_image_labels[i].pack(pady=i*6+5)

root.mainloop()
