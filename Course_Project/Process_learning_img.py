import os
from PIL import Image
import shutil

def convert_png_to_jpg(folder_path):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith("_final.png"):
            # Construct the new name by removing "_final"
            new_name = os.path.join(folder_path, filename.replace("_final.png", ".jpg"))
            old_path = os.path.join(folder_path, filename)

            # Open the PNG image and save it as a JPG image
            # with Image.open(old_path) as img:
            #     img.save(new_name, "JPEG")

            # Remove the old PNG file if needed
            os.remove(old_path)

    print("Conversion complete.")


def rearrange_folder_structure(folder_path):
    images_folder = os.path.join(folder_path, "images")
    target_folder = folder_path

    # Move all files from 'images' subfolder to the parent folder
    for filename in os.listdir(images_folder):
        source_path = os.path.join(images_folder, filename)
        target_path = os.path.join(target_folder, filename)
        shutil.move(source_path, target_path)

    # Remove the 'images' subfolder
    shutil.rmtree(images_folder)

    # Remove the 'index.html' file
    index_html_path = os.path.join(target_folder, "index.html")
    if os.path.exists(index_html_path):
        os.remove(index_html_path)

    print("Folder structure rearranged.")


def get_folder_names(parent_folder):
    # Get a list of all items in the parent folder
    items = os.listdir(parent_folder)
    # Filter out only folders
    # folder_names = [parent_folder+'/'+item for item in items]
    return items


for folder_name in get_folder_names("Dataset/UA-DETRAC/dehaze_learning"):
    folder_path = "Dataset/UA-DETRAC/dehaze_learning/" + folder_name
    print(folder_path)
    rearrange_folder_structure(folder_path)
    convert_png_to_jpg(folder_path)
