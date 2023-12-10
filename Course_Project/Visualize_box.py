import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')

def visualize_boxes(xml_file, image_path, frame_interval):
    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    frame_counter = 0  # Counter for frames

    # Iterate through frames in the XML file
    for frame in root.findall('.//frame'):
        if int(frame.get('num')) % frame_interval == 1:
            print(f"Frame: {frame.get('num')}")
            frame_counter += 1

            # Get image size
            image_file_path = image_path + "/img" + frame.get('num').zfill(5) + '.jpg'
            image = Image.open(image_file_path)

            # Get subplots
            fig, ax = plt.subplots(1)
            ax.imshow(image)

            targets = frame.find('.//target_list').findall('target')
            for target in targets:
                object_id = target.get('id')
                left = float(target.find('box').get('left'))
                top = float(target.find('box').get('top'))
                width = float(target.find('box').get('width'))
                height = float(target.find('box').get('height'))
                rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                # Add object_id and vehicle_type to the plot
                plt.text(left, top - 5, f"ID: {object_id}, Type: {target.find('attribute').get('vehicle_type')}", color='r')
            plt.savefig(f"Test_Output/MVI_39031_{frame.get('num')}.png")
            plt.close()

# Example usage

xml_file_modify = 'Dataset/UA-DETRAC-Annotations/test/MVI_39031.xml'
image_path_modify = 'Dataset/UA-DETRAC/gt/test/MVI_39031'
# visualize_boxes(xml_file, image_path, frame_interval=50)
visualize_boxes(xml_file_modify, image_path_modify, frame_interval=10)

