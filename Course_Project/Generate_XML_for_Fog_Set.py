import xml.etree.ElementTree as ET
import os
import re


def get_folder_names(parent_folder):
    # Get a list of all items in the parent folder
    items = os.listdir(parent_folder)
    # Filter out only folders
    # folder_names = [parent_folder+'/'+item for item in items]
    return items


def extract_numbers_from_filenames(folder_path):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"The specified folder path '{folder_path}' does not exist.")
        return

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Use regular expression to extract numbers from file names
    pattern = re.compile(r'img(\d+).jpg', re.IGNORECASE)

    # Extract numbers and remove leading zeros
    numbers = [int(match.group(1)) for file in files if (match := pattern.match(file))]

    return numbers


def filter_frames(xml_path, selected_frame_numbers, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    new_root = ET.Element("sequence", name=root.attrib["name"])
    sequence_attribute = root.find("sequence_attribute")
    if sequence_attribute is not None:
        new_root.append(ET.Element("sequence_attribute", attrib=sequence_attribute.attrib))

    ignored_region = root.find("ignored_region")
    if ignored_region is not None:
        new_root.append(ignored_region)

    for frame in root.findall("frame"):
        if int(frame.attrib["num"]) in selected_frame_numbers:
            new_root.append(frame)

    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_path, encoding="utf-8", xml_declaration=True)


# selected_frames = [51, 101, 151, 201, 251]
# input_path = 'Dataset/DETRAC-Test-Annotations-XML/MVI_39031.xml'
# output_path = "Truncate_MVI_39031.xml"
# filter_frames(input_path, selected_frames, output_path)

train_folder_path = "Dataset/UA-DETRAC/gt/train"
test_folder_path = "Dataset/UA-DETRAC/gt/test"
XML_train_folder_path = 'Dataset/DETRAC-Train-Annotations-XML'
XML_test_folder_path = 'Dataset/DETRAC-Test-Annotations-XML'
output_train_folder_path = 'Dataset/UA-DETRAC-Annotations/train'
output_test_folder_path = 'Dataset/UA-DETRAC-Annotations/test'

# Get the list of folder names
train_folder_list = get_folder_names(train_folder_path)
test_folder_list = get_folder_names(test_folder_path)

for train_folder in train_folder_list:
    # Get the list of frame numbers
    train_frame_numbers = extract_numbers_from_filenames(train_folder_path + '/' + train_folder)
    # Get the corresponding XML file
    xml_file = XML_train_folder_path + '/' + train_folder + '.xml'
    output_xml_file = output_train_folder_path + '/' + train_folder + '.xml'
    # Filter the XML files
    filter_frames(xml_file, train_frame_numbers, output_xml_file)


for test_folder in test_folder_list:
    # Get the list of frame numbers
    test_frame_numbers = extract_numbers_from_filenames(test_folder_path + '/' + test_folder)
    # Get the corresponding XML file
    xml_file = XML_test_folder_path + '/' + test_folder + '.xml'
    output_xml_file = output_test_folder_path + '/' + test_folder + '.xml'
    # Filter the XML files
    filter_frames(xml_file, test_frame_numbers, output_xml_file)
