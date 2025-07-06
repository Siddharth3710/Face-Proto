
# 🧠 Face Recognition System with Multi-Angle Detection

A production-grade, real-time face recognition system that captures **front, left, and right facial views** using a webcam.  
Built using **OpenCV**, **face_recognition**, and **Flask**, this system registers users, extracts embeddings, and performs  
live recognition using **cosine similarity**. Ideal for smart attendance, secure access control, or personalized UX.

---

## 🚀 Features

- 🎥 Real-time **Webcam Integration**
- 📸 Multi-Angle Registration (Front, Left, Right)
- 🗂️ Auto-save to `uploads/<username>/` directory
- 🧬 Embedding extraction via `face_recognition`
- 📈 Similarity matching using **Cosine Distance**
- 🔍 Live Face Recognition for returning users
- 🧪 Preview Before Save (quality control)
- 🌐 Lightweight Flask-based web interface

---

## 📁 Project Structure

```
face_recognition_project/
├── app.py                # Flask app entry point
├── capture.py            # Capture front/left/right face images
├── embeddings.py         # Extract/store facial embeddings
├── recognize.py          # Live recognition logic
├── utils.py              # Helper methods
├── uploads/              # User-specific image folders
│   └── <username>/
│       ├── front.jpg
│       ├── left.jpg
│       └── right.jpg
├── static/               # Frontend (CSS/JS/images)
├── templates/            # HTML (Jinja2)
├── embeddings.pkl        # Stored facial embeddings
└── requirements.txt      # Dependencies
```

---

## ⚙️ How It Works

### 👤 New User Registration
1. User enters a name on the interface.
2. Captures front, left, and right face images.
3. Saves them under `uploads/<username>/`.
4. Extracts 128D embeddings using `face_recognition`.
5. Stores embeddings in `embeddings.pkl`.

### 🧠 Existing User Recognition
1. System activates webcam and detects a face.
2. Extracts the embedding of the live face.
3. Matches against existing users via cosine similarity.
4. If similarity score > threshold → displays matched user.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

---





## 🤝 Contributing

Pull requests are welcome!  
If you have suggestions or want to improve the system, feel free to open an issue or submit a PR.

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 👋 About the Author

Developed with ❤️ by **[Siddharth Jha](https://github.com/yourusername)**  
Let’s connect and build something impactful together!
