#!/usr/bin/env python3
"""
JSONL File Creator for FormLens Training Data
Creates JSONL files with image paths and corresponding text responses.
"""

import json
import os
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional, Union
import shutil

class JSONLCreator:
    """Class to create JSONL files for FormLens training data."""
    
    def __init__(self, output_file: str = "training_data.jsonl"):
        """Initialize the JSONL creator."""
        self.output_file = output_file
        self.entries = []
    
    def add_entry(self, image_path: str, response: str, images: Optional[List[str]] = None):
        """Add a single entry to the JSONL data."""
        # Validate image path
        if not os.path.exists(image_path):
            print(f"Warning: Image file does not exist: {image_path}")
        
        # Create entry
        entry = {
            "query": f"<image>{image_path}",
            "response": response,
            "images": images if images is not None else [image_path]
        }
        
        self.entries.append(entry)
        return entry
    
    def add_entry_from_dict(self, entry_data: Dict):
        """Add entry from a dictionary."""
        required_fields = ["query", "response", "images"]
        for field in required_fields:
            if field not in entry_data:
                raise ValueError(f"Missing required field: {field}")
        
        self.entries.append(entry_data)
        return entry_data
    
    def load_from_csv(self, csv_file: str, image_path_column: str = "image_path", 
                     response_column: str = "response", delimiter: str = ","):
        """Load data from a CSV file."""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            
            for row_num, row in enumerate(reader, 1):
                try:
                    image_path = row[image_path_column]
                    response = row[response_column]
                    
                    # Create images list (can be single image or multiple)
                    images = [image_path]
                    if 'images' in row and row['images']:
                        # If images column exists, split by semicolon or comma
                        images = [img.strip() for img in row['images'].replace(';', ',').split(',') if img.strip()]
                    
                    self.add_entry(image_path, response, images)
                    
                except KeyError as e:
                    print(f"Warning: Missing column {e} in row {row_num}")
                except Exception as e:
                    print(f"Warning: Error processing row {row_num}: {e}")
    
    def load_from_txt_pairs(self, txt_file: str, image_dir: str = "", delimiter: str = "\t"):
        """Load data from a text file with image-response pairs."""
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Text file not found: {txt_file}")
        
        with open(txt_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split(delimiter)
                    if len(parts) < 2:
                        print(f"Warning: Invalid format in line {line_num}: {line}")
                        continue
                    
                    image_path = parts[0].strip()
                    response = parts[1].strip()
                    
                    # Add image directory if provided and path is relative
                    if image_dir and not os.path.isabs(image_path):
                        image_path = os.path.join(image_dir, image_path)
                    
                    self.add_entry(image_path, response)
                    
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
    
    def create_from_directory_structure(self, base_dir: str, response_mapping: Dict[str, str]):
        """Create entries from directory structure with response mapping."""
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    image_path = os.path.join(root, file)
                    
                    # Try to find response for this image
                    # First try exact filename match
                    filename = Path(file).stem
                    response = response_mapping.get(filename)
                    
                    # If not found, try with directory name
                    if not response:
                        dir_name = os.path.basename(root)
                        response = response_mapping.get(dir_name)
                    
                    # If still not found, use default or skip
                    if not response:
                        print(f"Warning: No response found for {image_path}")
                        continue
                    
                    self.add_entry(image_path, response)
    
    def save_jsonl(self, output_file: Optional[str] = None):
        """Save entries to JSONL file."""
        if output_file:
            self.output_file = output_file
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file) if os.path.dirname(self.output_file) else '.', exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as file:
            for entry in self.entries:
                json_line = json.dumps(entry, ensure_ascii=False)
                file.write(json_line + '\n')
        
        print(f"Saved {len(self.entries)} entries to {self.output_file}")
    
    def validate_entries(self) -> Dict[str, int]:
        """Validate all entries and return statistics."""
        stats = {
            'total_entries': len(self.entries),
            'valid_images': 0,
            'invalid_images': 0,
            'empty_responses': 0
        }
        
        for entry in self.entries:
            # Check image existence
            for image_path in entry['images']:
                if os.path.exists(image_path):
                    stats['valid_images'] += 1
                else:
                    stats['invalid_images'] += 1
            
            # Check response
            if not entry['response'].strip():
                stats['empty_responses'] += 1
        
        return stats
    
    def preview_entries(self, num_entries: int = 5):
        """Preview first few entries."""
        print(f"\nPreview of first {min(num_entries, len(self.entries))} entries:")
        print("=" * 80)
        
        for i, entry in enumerate(self.entries[:num_entries]):
            print(f"\nEntry {i+1}:")
            print(f"  Query: {entry['query'][:100]}{'...' if len(entry['query']) > 100 else ''}")
            print(f"  Response: {entry['response']}")
            print(f"  Images: {entry['images']}")
    
    def copy_images_to_output_dir(self, output_dir: str, copy_images: bool = True):
        """Copy images to output directory and update paths."""
        if not copy_images:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for entry in self.entries:
            new_images = []
            for image_path in entry['images']:
                if os.path.exists(image_path):
                    # Copy image to output directory
                    filename = os.path.basename(image_path)
                    new_path = os.path.join(output_dir, filename)
                    shutil.copy2(image_path, new_path)
                    new_images.append(new_path)
                else:
                    new_images.append(image_path)
            
            entry['images'] = new_images
            entry['query'] = f"<image>{new_images[0]}"

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Create JSONL files for FormLens training data')
    parser.add_argument('--output', '-o', default='training_data.jsonl', help='Output JSONL file')
    parser.add_argument('--csv', help='Input CSV file with image paths and responses')
    parser.add_argument('--txt', help='Input text file with image-response pairs')
    parser.add_argument('--image_dir', default='', help='Base directory for images (for CSV/TXT input)')
    parser.add_argument('--image_path_column', default='image_path', help='CSV column name for image paths')
    parser.add_argument('--response_column', default='response', help='CSV column name for responses')
    parser.add_argument('--delimiter', default='\t', help='Delimiter for text files')
    parser.add_argument('--copy_images', action='store_true', help='Copy images to output directory')
    parser.add_argument('--output_dir', help='Output directory for copied images')
    parser.add_argument('--validate', action='store_true', help='Validate entries before saving')
    parser.add_argument('--preview', action='store_true', help='Preview entries before saving')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode for manual entry')
    
    args = parser.parse_args()
    
    # Initialize JSONL creator
    creator = JSONLCreator(args.output)
    
    # Load data from different sources
    if args.csv:
        print(f"Loading data from CSV: {args.csv}")
        creator.load_from_csv(args.csv, args.image_path_column, args.response_column)
    elif args.txt:
        print(f"Loading data from text file: {args.txt}")
        creator.load_from_txt_pairs(args.txt, args.image_dir, args.delimiter)
    elif args.interactive:
        print("Interactive mode - Enter image paths and responses (type 'done' to finish)")
        while True:
            image_path = input("Image path: ").strip()
            if image_path.lower() == 'done':
                break
            
            response = input("Response: ").strip()
            if not response:
                print("Response cannot be empty!")
                continue
            
            creator.add_entry(image_path, response)
            print(f"Added entry: {image_path} -> {response}")
    else:
        # Example usage
        print("No input specified. Creating example entries...")
        creator.add_entry("/path/to/image1.jpg", "Sample response 1")
        creator.add_entry("/path/to/image2.jpg", "Sample response 2")
    
    # Copy images if requested
    if args.copy_images and args.output_dir:
        print(f"Copying images to: {args.output_dir}")
        creator.copy_images_to_output_dir(args.output_dir, copy_images=True)
    
    # Validate entries
    if args.validate:
        stats = creator.validate_entries()
        print(f"\nValidation Results:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Valid images: {stats['valid_images']}")
        print(f"  Invalid images: {stats['invalid_images']}")
        print(f"  Empty responses: {stats['empty_responses']}")
    
    # Preview entries
    if args.preview:
        creator.preview_entries()
    
    # Save JSONL file
    creator.save_jsonl()

if __name__ == "__main__":
    main()
