import json
import os
import zipfile
from tqdm import tqdm

# PFADE ANPASSEN
ZIP_PATH = "/mnt/c/Users/rapha/Documents/projects/doclayout-detection/DocLayNet_core.zip"
EXTRACTED_COCO_DIR = ".data/COCO"
OUTPUT_DIR = ".data/images"

TARGET_CATEGORIES = ["scientific_articles"]

SPLIT_CONFIGS = [
    {"json_file": "train.json", "yolo_split": "train", "max_samples": 5000},
    {"json_file": "val.json", "yolo_split": "val", "max_samples": 1000},
    {"json_file": "test.json", "yolo_split": "test", "max_samples": 1000}
]

# 1. Zielordner nur für die Bilder erstellen
for config in SPLIT_CONFIGS:
    os.makedirs(f"{OUTPUT_DIR}/{config['yolo_split']}", exist_ok=True)

# 2. Bilder extrahieren
with zipfile.ZipFile(ZIP_PATH, 'r') as archive:
    
    for config in SPLIT_CONFIGS:
        json_path = os.path.join(EXTRACTED_COCO_DIR, config["json_file"])
        print(f"\nExtrahiere Bilder für {config['json_file']}...")
        
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
            
        count = 0
        for img_info in tqdm(coco_data['images']):
            cat_name = img_info.get("doc_category")
            
            if cat_name in TARGET_CATEGORIES:
                file_name = img_info["file_name"]
                img_id = img_info["id"]
                
                try:
                    # Pfad innerhalb der ZIP-Datei ansteuern
                    zip_img_path = f"PNG/{os.path.basename(file_name)}"
                    img_data = archive.read(zip_img_path)
                    
                    # Bild direkt im Zielordner speichern
                    output_path = f"{OUTPUT_DIR}/{config['yolo_split']}/{img_id}.png"
                    with open(output_path, "wb") as img_f:
                        img_f.write(img_data)
                        
                    count += 1
                    if count >= config["max_samples"]:
                        break
                except KeyError:
                    # Falls ein Bild in der ZIP fehlen sollte, einfach überspringen
                    continue

print(f"\nErfolgreich! Alle Bilder wurden nach '{OUTPUT_DIR}' extrahiert.")