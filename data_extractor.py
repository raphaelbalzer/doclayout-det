import json
import os
import zipfile
from tqdm import tqdm
import random


MULTICATEGORY = True
if MULTICATEGORY:
    path_suffix = "multi_cat"
else:
    path_suffix = "single_cat"

if MULTICATEGORY:
    TARGET_CATEGORIES = ["financial_reports", "scientific_articles", "laws_and_regulations"]
    SPLIT_CONFIGS = [
        {"json_file": "train.json", "yolo_split": "train", "samples_per_category": 3000},
        {"json_file": "val.json", "yolo_split": "val", "samples_per_category": 944},
        {"json_file": "test.json", "yolo_split": "test", "samples_per_category": 784}
    ]
else:
    TARGET_CATEGORIES = ["scientific_articles"]
    SPLIT_CONFIGS = [
        {"json_file": "train.json", "yolo_split": "train", "samples_per_category": 5000},
        {"json_file": "val.json", "yolo_split": "val", "samples_per_category": 1000},
        {"json_file": "test.json", "yolo_split": "test", "samples_per_category": 1000}
    ]

ZIP_PATH = "/mnt/c/Users/rapha/Documents/projects/doclayout-detection/DocLayNet_core.zip"
EXTRACTED_COCO_DIR = ".coco_labels_full"
OUTPUT_DIR = ".data/images"

random.seed(42)

with zipfile.ZipFile(ZIP_PATH, 'r') as archive:
    
    for config in SPLIT_CONFIGS:
        json_path = os.path.join(EXTRACTED_COCO_DIR, config["json_file"])
        print(f"\Processing {config['json_file']} (Mode: {'Multi' if MULTICATEGORY else 'Single'})...")
        
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
            
        selected_images = []
        
        for category in TARGET_CATEGORIES:
            cat_images = [img for img in coco_data['images'] if img.get("doc_category") == category]
            
            random.shuffle(cat_images)
            
            # Limit to the specified number of samples per category
            cat_subset = cat_images[:config["samples_per_category"]]
            selected_images.extend(cat_subset)
            
            print(f"  Category '{category}': {len(cat_subset)} Samples selected.")

        # Shuffle again after collecting from all categories to mix them up
        random.shuffle(selected_images)
        
        # create output directory for images
        os.makedirs(f"{OUTPUT_DIR}/{config['yolo_split']}", exist_ok=True)
        
        # 3. Bilder extrahieren
        print(f"Extrahiere insgesamt {len(selected_images)} Bilder...")
        actual_saved_images = []
        
        for img_info in tqdm(selected_images):
            file_name = img_info["file_name"]
            
            try:
                zip_img_path = f"PNG/{os.path.basename(file_name)}"
                img_data = archive.read(zip_img_path)
                
                output_path = f"{OUTPUT_DIR}/{config['yolo_split']}/{os.path.basename(file_name)}"
                with open(output_path, "wb") as img_f:
                    img_f.write(img_data)
                
                actual_saved_images.append(img_info)
            except KeyError:
                continue
                
        final_image_ids = {img['id'] for img in actual_saved_images}
        
        # 4. COCO-JSON auf die extrahierten Bilder und deren Annotations kürzen
        print("Erstelle gekürztes COCO-JSON...")
        filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in final_image_ids]
        
        subset_coco = {
            "categories": coco_data["categories"],
            "images": actual_saved_images,
            "annotations": filtered_annotations
        }
        
        os.makedirs(f".data/labels_coco_{path_suffix}", exist_ok=True)
        with open(f".data/labels_coco_{path_suffix}/labels_coco_{config['yolo_split']}.json", "w") as out_json:
            json.dump(subset_coco, out_json)

print(f"\nFertig! Daten erfolgreich partitioniert unter '{OUTPUT_DIR}' und '.data/labels_coco_{path_suffix}/'")