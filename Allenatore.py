from ultralytics import YOLO
modello = YOLO("yolo11n.pt")
modello.train(data="data.yaml", epochs=15, verbose=True)
