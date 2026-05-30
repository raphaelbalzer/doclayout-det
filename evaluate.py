from ultralytics import YOLO

def main():
    # Das beste, trainierte Modell aus dem letzten Training laden
    # Passe den Pfad an, falls dein aktuellstes Training in einem anderen Ordner liegt (z.B. train-3)
    model = YOLO("runs/detect/train-2/weights/best.pt")

    print("Starte Evaluierung auf dem Validation-Datensatz...")
    
    # Model evaluieren. model.val() nutzt automatisch die in den args.yaml gespeicherten config
    metrics = model.val()

    # Übersichtliche Ausgabe der wichtigsten Metriken
    print("\n--- Evaluierungsergebnisse ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")

if __name__ == "__main__":
    main()
