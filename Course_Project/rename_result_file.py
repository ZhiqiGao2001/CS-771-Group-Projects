import os

def rename_files(base_folder, file_name):
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file == file_name:
                folder_name = os.path.basename(root)
                subfolder_name = os.path.basename(os.path.dirname(root))
                new_name = f"{subfolder_name}_{folder_name}_{file}"

                # Full paths for old and new names
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_name)
                # Rename the file
                os.rename(old_path, new_path)
                # print(f"Renamed: {old_path} to {new_path}")

# Replace 'path/to/your/folder' with the actual path to your main folder
folder_path = 'C:/Users/ZhiQi/OneDrive/桌面/yolov5/runs/detect/yolov8m_gt'
rename_files(folder_path, "confusion_matrix_normalized.png")
