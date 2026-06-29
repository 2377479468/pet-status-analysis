from ultralytics import YOLO

def main():
    model = YOLO("yolov8m-pose.pt")

    results = model.train(
        data="dog-pose.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        optimizer="auto",
        patience=25,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.0,
        mosaic=0.2,

        workers=0,
        cache = False,
        verbose = True,
        project="../runs/dog-pose",
        name="yolov8m_100e_640",
        exist_ok=True
    )

    print(results)

if __name__ == "__main__":
    main()