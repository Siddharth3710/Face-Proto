import os

path = "embeddings.pkl"

if os.path.exists(path):
    os.remove(path)
    print("🗑️ embeddings.pkl has been deleted (reset successful).")
else:
    print("⚠️ embeddings.pkl does not exist.")
