import json
import xml.etree.ElementTree as ET

from torch.utils import data

def read_json(file_path):
    """
    Read and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file to read
        
    Returns:
        dict: Parsed JSON data as a Python dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        IOError: If there's an error reading the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e.msg}", e.doc, e.pos)
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")

def write_json(data, file_path):
    """
    Write data to a JSON file.
    
    Args:
        data (dict): Data to write to the JSON file
        file_path (str): Path to the JSON file to write
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
    except IOError as e:
        raise IOError(f"Error writing to file {file_path}: {str(e)}")

def read_eaf(file_path):
    """
    Read and parse an EAF file (ELAN Annotation Format) and convert it to JSON.
    
    Args:
        file_path (str): Path to the EAF file to read
        
    Returns:
        dict: Parsed EAF data as a Python dictionary following the template structure
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ET.ParseError: If the file contains invalid XML
        IOError: If there's an error reading the file
    """
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Initialize the result dictionary
        result = {
            "audio_url": "",
            "last_used_annotation_id": 0,
            "tiers": []
        }
        
        # Extract audio URL from MEDIA_DESCRIPTOR
        media_descriptor = root.find('.//MEDIA_DESCRIPTOR')
        if media_descriptor is not None:
            result["audio_url"] = media_descriptor.get('MEDIA_URL', '')
        
        # Extract last used annotation ID
        last_used_prop = root.find('.//PROPERTY[@NAME="lastUsedAnnotationId"]')
        if last_used_prop is not None:
            result["last_used_annotation_id"] = int(last_used_prop.text or 0)
        
        # Create time slot mapping
        time_slots = {}
        for time_slot in root.findall('.//TIME_SLOT'):
            slot_id = time_slot.get('TIME_SLOT_ID')
            time_value = time_slot.get('TIME_VALUE')
            if slot_id and time_value:
                time_slots[slot_id] = time_value
        
        # Process tiers
        tiers = root.findall('.//TIER')
        
        for tier in tiers:
            tier_id = tier.get('TIER_ID', '')
            if not tier_id:
                continue
                
            # Add tier to the result using the actual tier ID as key
            result[tier_id] = {"id": tier_id, "annotations": []}
            result["tiers"].append(tier_id)
            
            # Process annotations in this tier
            annotations = tier.findall('.//ALIGNABLE_ANNOTATION')
            for annotation in annotations:
                ann_id = annotation.get('ANNOTATION_ID', '')
                time_ref1 = annotation.get('TIME_SLOT_REF1', '')
                time_ref2 = annotation.get('TIME_SLOT_REF2', '')
                ann_value = annotation.find('ANNOTATION_VALUE')
                
                if ann_id and time_ref1 and time_ref2 and ann_value is not None:
                    start_time = time_slots.get(time_ref1, '0')
                    end_time = time_slots.get(time_ref2, '0')
                    value = ann_value.text or ''
                    
                    result[tier_id]["annotations"].append({
                        "id": ann_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "value": value
                    })
        
        return result
        
    except FileNotFoundError:
        raise FileNotFoundError(f"EAF file not found: {file_path}")
    except ET.ParseError as e:
        raise ET.ParseError(f"Invalid XML in EAF file {file_path}: {e.msg}")
    except IOError as e:
        raise IOError(f"Error reading EAF file {file_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error parsing EAF file {file_path}: {str(e)}")

def write_eaf(data, file_path):
    """
    Write JSON data to an EAF file (ELAN Annotation Format).
    
    Args:
        data (dict): JSON data to convert to EAF format
        file_path (str): Path to the EAF file to write
        
    Raises:
        IOError: If there's an error writing the file
        KeyError: If required data fields are missing
    """
    try:
        # Create the root element
        root = ET.Element("ANNOTATION_DOCUMENT")
        root.set("AUTHOR", "")
        root.set("DATE", "2025-09-05T11:03:06-05:00")
        root.set("FORMAT", "3.0")
        root.set("VERSION", "3.0")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://www.mpi.nl/tools/elan/EAFv3.0.xsd")
        
        # Create HEADER
        header = ET.SubElement(root, "HEADER")
        header.set("MEDIA_FILE", "")
        header.set("TIME_UNITS", "milliseconds")
        
        # Create MEDIA_DESCRIPTOR
        media_desc = ET.SubElement(header, "MEDIA_DESCRIPTOR")
        media_desc.set("MEDIA_URL", data.get("audio_url", ""))
        media_desc.set("MIME_TYPE", "audio/x-wav")
        media_desc.set("RELATIVE_MEDIA_URL", "./audio.wav")
        
        # Create PROPERTY elements
        urn_prop = ET.SubElement(header, "PROPERTY")
        urn_prop.set("NAME", "URN")
        urn_prop.text = "urn:nl-mpi-tools-elan-eaf:353af8d6-c1fc-4cdb-9716-602eb20d5179"
        
        last_used_prop = ET.SubElement(header, "PROPERTY")
        last_used_prop.set("NAME", "lastUsedAnnotationId")
        last_used_prop.text = str(data.get("last_used_annotation_id", 0))
        
        # Create TIME_ORDER
        time_order = ET.SubElement(root, "TIME_ORDER")
        
        # Collect all time values from annotations
        time_values = set()
        for tier_id in data.get("tiers", []):
            if tier_id in data:
                for annotation in data[tier_id].get("annotations", []):
                    time_values.add(annotation.get("start_time", "0"))
                    time_values.add(annotation.get("end_time", "0"))
        
        # Create time slots
        time_slots = {}
        for i, time_val in enumerate(sorted(time_values), 1):
            slot_id = f"ts{i}"
            time_slots[time_val] = slot_id
            time_slot = ET.SubElement(time_order, "TIME_SLOT")
            time_slot.set("TIME_SLOT_ID", slot_id)
            time_slot.set("TIME_VALUE", time_val)
        
        # Create TIER elements
        for tier_id in data.get("tiers", []):
            if tier_id in data:
                tier = ET.SubElement(root, "TIER")
                tier.set("LINGUISTIC_TYPE_REF", "default-lt")
                tier.set("TIER_ID", tier_id)
                
                # Create ANNOTATION elements for this tier
                for annotation in data[tier_id].get("annotations", []):
                    ann_elem = ET.SubElement(tier, "ANNOTATION")
                    align_ann = ET.SubElement(ann_elem, "ALIGNABLE_ANNOTATION")
                    align_ann.set("ANNOTATION_ID", annotation.get("id", ""))
                    align_ann.set("TIME_SLOT_REF1", time_slots.get(annotation.get("start_time", "0"), "ts1"))
                    align_ann.set("TIME_SLOT_REF2", time_slots.get(annotation.get("end_time", "0"), "ts2"))
                    
                    ann_value = ET.SubElement(align_ann, "ANNOTATION_VALUE")
                    ann_value.text = annotation.get("value", "")
        
        # Create LINGUISTIC_TYPE
        ling_type = ET.SubElement(root, "LINGUISTIC_TYPE")
        ling_type.set("GRAPHIC_REFERENCES", "false")
        ling_type.set("LINGUISTIC_TYPE_ID", "default-lt")
        ling_type.set("TIME_ALIGNABLE", "true")
        
        # Create CONSTRAINT elements
        constraints = [
            ("Time subdivision of parent annotation's time interval, no time gaps allowed within this interval", "Time_Subdivision"),
            ("Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered", "Symbolic_Subdivision"),
            ("1-1 association with a parent annotation", "Symbolic_Association"),
            ("Time alignable annotations within the parent annotation's time interval, gaps are allowed", "Included_In")
        ]
        
        for desc, stereo in constraints:
            constraint = ET.SubElement(root, "CONSTRAINT")
            constraint.set("DESCRIPTION", desc)
            constraint.set("STEREOTYPE", stereo)
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ", level=0)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
        
    except IOError as e:
        raise IOError(f"Error writing EAF file {file_path}: {str(e)}")
    except KeyError as e:
        raise KeyError(f"Missing required data field: {e}")
    except Exception as e:
        raise Exception(f"Error creating EAF file {file_path}: {str(e)}")

def remove_annotation_by_id(data, annotation_id):
    """
    Remove an annotation by its ID from the data structure.
    
    Args:
        data (dict): The data structure containing annotations
        annotation_id (str): The ID of the annotation to remove
        
    Returns:
        bool: True if annotation was found and removed, False otherwise
    """
    # Search through all tiers for the annotation
    for tier_id in data.get("tiers", []):
        if tier_id in data and "annotations" in data[tier_id]:
            annotations = data[tier_id]["annotations"]
            
            # Find and remove the annotation with matching ID
            for i, annotation in enumerate(annotations):
                if annotation.get("id") == annotation_id:
                    # Remove the annotation
                    annotations.pop(i)
                    
                    # Update last_used_annotation_id if necessary
                    # Extract numeric part from annotation ID (e.g., "a123" -> 123)
                    try:
                        if annotation_id.startswith("a"):
                            removed_id_num = int(annotation_id[1:])
                            current_last_id = data.get("last_used_annotation_id", 0)
                            
                            # If we removed the highest ID, we need to find the new highest
                            if removed_id_num == current_last_id:
                                max_id = 0
                                for tier in data.get("tiers", []):
                                    if tier in data and "annotations" in data[tier]:
                                        for ann in data[tier]["annotations"]:
                                            ann_id = ann.get("id", "")
                                            if ann_id.startswith("a"):
                                                try:
                                                    ann_num = int(ann_id[1:])
                                                    max_id = max(max_id, ann_num)
                                                except ValueError:
                                                    pass
                                data["last_used_annotation_id"] = max_id
                    except (ValueError, IndexError):
                        # If annotation ID format is unexpected, just leave last_used_annotation_id as is
                        pass
                    
                    return True
    
    return False

def remove_short_annotations(data, min_chunk_duration):
    """
    Remove annotations with a duration less than the specified minimum duration.
    
    Args:
        data (dict): The data structure containing annotations
        min_chunk_duration (int): The minimum duration in milliseconds
        
    Returns:
        dict: Statistics about the removal operation including:
            - removed_count: Number of annotations removed
            - remaining_count: Number of annotations remaining
            - affected_tiers: List of tier IDs that had annotations removed
    """
    removed_count = 0
    affected_tiers = set()
    removed_annotation_ids = []
    
    # Process each tier
    for tier_id in data.get("tiers", []):
        if tier_id in data and "annotations" in data[tier_id]:
            annotations = data[tier_id]["annotations"]
            annotations_to_remove = []
            
            # Find annotations that are too short
            for annotation in annotations:
                try:
                    start_time = int(annotation.get("start_time", "0"))
                    end_time = int(annotation.get("end_time", "0"))
                    duration = end_time - start_time
                    
                    if duration < min_chunk_duration:
                        annotations_to_remove.append(annotation)
                        removed_annotation_ids.append(annotation.get("id", ""))
                        affected_tiers.add(tier_id)
                except (ValueError, TypeError):
                    # Skip annotations with invalid time values
                    continue
            
            # Remove the short annotations
            for annotation in annotations_to_remove:
                if annotation in annotations:
                    annotations.remove(annotation)
                    removed_count += 1
    
    # Update last_used_annotation_id if we removed annotations
    if removed_annotation_ids:
        # Find the highest remaining annotation ID
        max_id = 0
        for tier in data.get("tiers", []):
            if tier in data and "annotations" in data[tier]:
                for ann in data[tier]["annotations"]:
                    ann_id = ann.get("id", "")
                    if ann_id.startswith("a"):
                        try:
                            ann_num = int(ann_id[1:])
                            max_id = max(max_id, ann_num)
                        except ValueError:
                            pass
        
        # Update last_used_annotation_id
        data["last_used_annotation_id"] = max_id
        
        # Clean up empty tiers (optional - remove if you want to keep empty tiers)
        tiers_to_remove = []
        for tier_id in data.get("tiers", []):
            if tier_id in data and "annotations" in data[tier_id]:
                if not data[tier_id]["annotations"]:
                    tiers_to_remove.append(tier_id)
        
        for tier_id in tiers_to_remove:
            if tier_id in data:
                del data[tier_id]
            if tier_id in data.get("tiers", []):
                data["tiers"].remove(tier_id)
    
    # Calculate remaining count
    remaining_count = 0
    for tier in data.get("tiers", []):
        if tier in data and "annotations" in data[tier]:
            remaining_count += len(data[tier]["annotations"])
    
    return {
        "removed_count": removed_count,
        "remaining_count": remaining_count,
        "affected_tiers": list(affected_tiers),
        "removed_annotation_ids": removed_annotation_ids
    }

if __name__ == "__main__":
    data = read_eaf("templates/test.eaf")
    write_json(data, "templates/test.json")