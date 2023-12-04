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
            # os.remove(old_path)

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


def modify_resolution(folder_path):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        try:
            # Open the image
            with Image.open(filename) as img:
                # Resize the image
                resized_img = img.resize((960, 540), Image.ANTIALIAS)

                # Save the resized image
                resized_img.save(filename)
        except Exception as e:
            print(f"Error: {e}")


def move_folders_to_test(big_folder_path, folder_names_to_move):
    test_folder_name = "test"

    # Create a test folder if it doesn't exist
    test_folder_path = os.path.join(big_folder_path, test_folder_name)
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)

    # Iterate through the subfolders of the big folder
    for subfolder_name in os.listdir(big_folder_path):
        subfolder_path = os.path.join(big_folder_path, subfolder_name)

        # Check if it's a directory and its name is in the list
        if os.path.isdir(subfolder_path) and subfolder_name in folder_names_to_move:
            # Move the subfolder to the test folder
            new_location = os.path.join(test_folder_path, subfolder_name)
            shutil.move(subfolder_path, new_location)
            print(f"Moved '{subfolder_name}' to '{test_folder_name}' folder.")


# for folder_name in get_folder_names("Dataset/UA-DETRAC/dehaze_learning"):
#     folder_path = "Dataset/UA-DETRAC/dehaze_learning/" + folder_name
#     print(folder_path)
    # rearrange_folder_structure(folder_path)
    # convert_png_to_jpg(folder_path)
# move_folders_to_test("Dataset/UA-DETRAC/dehaze_learning",get_folder_names("Dataset/UA-DETRAC/dehaze_DarkChannel/test"))
