import os
import shutil

def organize_folders(main_folder_path):
    # List all subfolders in the main folder
    subfolders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]

    # Create four folders for different suffixes
    for suffix in ["_0.01", "_0.005", "_0.02", "_0.03"]:
        new_folder_path = os.path.join(main_folder_path, f"folder{suffix}")
        os.makedirs(new_folder_path, exist_ok=True)

        # Move folders with the respective suffix to the new folder
        for folder in subfolders:
            if folder.endswith(suffix):
                old_folder_path = os.path.join(main_folder_path, folder)
                shutil.move(old_folder_path, new_folder_path)

    print("Folders organized.")

# Example usage:
main_folder_path = "Dataset/UA-DETRAC/hazy/test"
organize_folders(main_folder_path)