from ultralytics import YOLO

model=YOLO(r"runs\detect\train\weights\best.pt")

model.predict(source=r"fire3.mp4", imgsz=640, conf=0.7, show=True, save=True)