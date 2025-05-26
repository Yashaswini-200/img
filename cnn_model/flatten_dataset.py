import os
import shutil

def flatten_images(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(folder, file)
                if src_path != dst_path:
                    shutil.move(src_path, dst_path)
    for dirpath, dirnames, filenames in os.walk(folder, topdown=False):
        if dirpath != folder and not filenames:
            os.rmdir(dirpath)

# Flatten AI and Real folders
flatten_images('training_data/AI')
flatten_images('training_data/Real')

print("✅ All images flattened. You’re ready to train like a beast.")
