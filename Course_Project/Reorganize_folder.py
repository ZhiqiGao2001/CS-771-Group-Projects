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


def rename_subfolders(big_folder_path):
    # Iterate through the subfolders of the big folder
    for subfolder_name in os.listdir(big_folder_path):
        subfolder_path = os.path.join(big_folder_path, subfolder_name)

        # Check if it's a directory and the name matches the specified pattern
        for sub_name in os.listdir(subfolder_path):
            # Extract the desired part before the second "_"
            extract_name = sub_name.split("_", 2)[:2]
            new_name = "_".join(extract_name)
            os.rename(os.path.join(subfolder_path, sub_name), os.path.join(subfolder_path, new_name))
            print(f"Renamed '{sub_name}' to '{new_name}'.")


# Example usage:
# big_folder_path = "Dataset/UA-DETRAC/dehaze_ColorCorrection/test"
# rename_subfolders(big_folder_path)


# main_folder_path = "Dataset/UA-DETRAC/dehaze_learning/test"
# organize_folders(main_folder_path)
