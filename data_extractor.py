import json
import os
import zipfile
from tqdm import tqdm
import random

# PFADE ANPASSEN
ZIP_PATH = "/mnt/c/Users/rapha/Documents/projects/doclayout-detection/DocLayNet_core.zip"
EXTRACTED_COCO_DIR = ".coco_labels_full"
OUTPUT_DIR = ".data/images"

TARGET_CATEGORIES = ["scientific_articles"]

SPLIT_CONFIGS = [
    {"json_file": "train.json", "yolo_split": "train", "max_samples": 5000},
    {"json_file": "val.json", "yolo_split": "val", "max_samples": 1000},
    {"json_file": "test.json", "yolo_split": "test", "max_samples": 1000}
]

# Seed setzen für Reproduzierbarkeit (wichtig für die Wissenschaft!)
random.seed(42)

with zipfile.ZipFile(ZIP_PATH, 'r') as archive:
    
    for config in SPLIT_CONFIGS:
        json_path = os.path.join(EXTRACTED_COCO_DIR, config["json_file"])
        print(f"\nVerarbeite {config['json_file']}...")
        
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
            
        # 1. Filtere alle Bilder, die zu unseren Wunschkategorien gehören
        filtered_images = [img for img in coco_data['images'] if img.get("doc_category") in TARGET_CATEGORIES]
        
        # 2. JETZT SHUFFELN WIR DIE GEFILTERTEN BILDER
        random.shuffle(filtered_images)
        
        # 3. Kürze die Liste auf die gewünschte Subset-Größe
        selected_images = filtered_images[:config["max_samples"]]
        selected_image_ids = {img['id'] for img in selected_images}
        
        # 4. Erstelle die Ordnerstruktur für die Bilder
        os.makedirs(f"{OUTPUT_DIR}/{config['yolo_split']}", exist_ok=True)
        
        # 5. Bilder extrahieren
        print(f"Extrahiere {len(selected_images)} geshuffelte Bilder...")
        actual_saved_images = []
        
        for img_info in tqdm(selected_images):
            file_name = img_info["file_name"]
            img_id = img_info["id"]
            
            try:
                zip_img_path = f"PNG/{os.path.basename(file_name)}"
                img_data = archive.read(zip_img_path)
                
                output_path = f"{OUTPUT_DIR}/{config['yolo_split']}/{img_id}.png"
                with open(output_path, "wb") as img_f:
                    img_f.write(img_data)
                
                actual_saved_images.append(img_info)
            except KeyError:
                continue
                
        # Update die ID-Liste, falls doch mal ein Bild in der ZIP fehlte
        final_image_ids = {img['id'] for img in actual_saved_images}
        
        # 6. Jetzt kürzen wir das COCO-JSON auf genau diese Bilder gekoppelt mit ihren Annotations
        print("Erstelle gekürztes COCO-JSON...")
        filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in final_image_ids]
        
        # Das neue JSON-Objekt bauen (Metadaten und Kategorien bleiben erhalten)
        subset_coco = {
            "categories": coco_data["categories"],
            "images": actual_saved_images,
            "annotations": filtered_annotations
        }
        
        # Speicher das kleine, feine JSON direkt im jeweiligen Split-Ordner ab
        os.makedirs(f".data/labels_coco", exist_ok=True)
        with open(f".data/labels_coco/labels_coco_{config['yolo_split']}.json", "w") as out_json:
            json.dump(subset_coco, out_json)

print(f"\nFertig! Bilder extrahiert und perfekt gekürzte JSONs erstellt unter '{OUTPUT_DIR}'")