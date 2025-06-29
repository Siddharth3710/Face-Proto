from capture_face import capture_and_save
from face_helper import get_face_embedding

if __name__ == "__main__":
    file = capture_and_save("front")
    emb = get_face_embedding(file)

    if emb:
        print("[✅] Face embedding generated successfully.")
    else:
        print("[❌] Failed to generate embedding.")
