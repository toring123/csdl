import mysql.connector
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import dlib
import KDTree
# 🔹 Khởi tạo detector & predictor của dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Đường dẫn đến file model

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
# 🔹 Hàm trích xuất đặc trưng từ ảnh
def extract_facial_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        return None

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(pt.x, pt.y) for pt in shape.parts()]

        p36, p45 = landmarks[36], landmarks[45] # khoảng cách 2 mắt
        p48, p54 = landmarks[48], landmarks[54] # độ dài miệng
        p31, p35 = landmarks[31], landmarks[35] # chiều ngang mũi
        p27, p33 = landmarks[27], landmarks[30] # độ dài sống mũi
        p3, p13 = landmarks[3], landmarks[13] # độ rộng hàm
        p17, p21 = landmarks[17], landmarks[21] # độ dài cung mày trái
        p8, p27 = landmarks[8], landmarks[27] # chiều cao khuôn mặt

        # Tính các khoảng cách
        d_eye = euclidean(p36, p45)
        d_mouth = euclidean(p48, p54)
        d_nose = euclidean(p31, p35)
        d_jaw = euclidean(p3, p13)
        d_face_height = euclidean(p8, p27)

        # Tránh chia cho 0
        if d_face_height == 0 or d_mouth == 0:
            return None

        # Tính các đặc trưng
        features = [
            d_eye / d_face_height,
            d_mouth / d_face_height,
            d_nose / d_face_height,
            d_jaw / d_face_height,
        ]

        return features

# 🔹 Hàm tải dữ liệu từ MySQL
def load_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="csdldpt"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT image_path, d_eye_norm, d_mouth_norm, d_nose_norm, d_jaw_norm from face_features")
    
    rows = cursor.fetchall()
    paths = [row[0] for row in rows]
    data = np.array([row[1:] for row in rows])

    cursor.close()
    conn.close()
    return paths, data

# 🔹 Hàm tìm ảnh gần nhất
def find_nearest_image(query_point, k):
    results = KDTree.k_nearest(kd_tree, query_point, k)

    # Hiển thị ảnh gần nhất
    i = 0
    for dist, path, _ in results:
        print(dist, path)
        img = Image.open(path).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        result_image_labels[i].config(image=img)
        result_image_labels[i].image = img
        result_labels[i].config(text=f"Ảnh gần thứ {i+1}: {path}\nKhoảng cách: {dist:.5f}")
        i = i +1

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

points = list(zip(paths, data))

# print(data)
# std_devs = np.std(data, axis=0)
# print(std_devs)

kd_tree = KDTree.build_kdtree(points)

# giao diện
root = tk.Tk()
root.title("Tìm kiếm ảnh bằng KD-tree")

btn_choose = tk.Button(root, text="Chọn ảnh", command=choose_image)  # Giả sử hàm choose_image đã định nghĩa
btn_choose.pack(pady=10)

input_image_label = tk.Label(root)
input_image_label.pack(pady=5)

# Tạo một frame để chứa các kết quả theo chiều ngang
result_frame = tk.Frame(root)
result_frame.pack(pady=10)

result_labels = []
result_image_labels = []

for i in range(3):
    item_frame = tk.Frame(result_frame)
    item_frame.pack(side="left", padx=10)  # Các item nằm ngang

    label = tk.Label(item_frame, text=f"Ảnh {i+1}", fg="blue")
    label.pack()

    img_label = tk.Label(item_frame)
    img_label.pack()

    result_labels.append(label)
    result_image_labels.append(img_label)

root.mainloop()
