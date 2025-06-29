from deepface import DeepFace

def get_face_embedding(img_path):
    try:
        print(f"🔍 Generating embedding using SFace + OpenCV for: {img_path}")
        obj = DeepFace.represent(
            img_path=img_path,
            model_name="SFace",  # or "Facenet"
            detector_backend="opencv",  # use opencv or mtcnn for speed
            enforce_detection=False  # ✅ CRITICAL for side views!
        )
        if obj and 'embedding' in obj[0]:
            return obj[0]['embedding']
        else:
            print("⚠️ No embedding returned")
            return None
    except Exception as e:
        print(f"❌ Error in get_face_embedding(): {e}")
        return None
