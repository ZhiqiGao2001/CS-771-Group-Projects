import xml.etree.ElementTree as ET


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

# Example: Keep frames with numbers 1, 3, and 5
selected_frames = [51, 101, 151, 201, 251]
input_path = 'Dataset/DETRAC-Test-Annotations-XML/MVI_39031.xml'
output_path = "Truncate_MVI_39031.xml"
filter_frames(input_path, selected_frames, output_path)
