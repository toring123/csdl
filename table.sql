Drop table if exists face_features;
CREATE TABLE face_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(255),          -- Đường dẫn ảnh

    d_eye_norm FLOAT,
    d_mouth_norm FLOAT,
    d_nose_norm FLOAT,
    d_jaw_norm FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
