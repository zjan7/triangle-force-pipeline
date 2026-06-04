import cv2
from pypylon import pylon
from pathlib import Path
import shutil
import traceback

from src.real_visual_pipeline import run_real_visual_pipeline
from src.real_inference_writer import write_real_inference_sample


PROJECT_DIR = Path(r"C:\Users\zibej\Documents\beppy")

INPUT_DIR = PROJECT_DIR / "inputs"
DEFORMED_DIR = INPUT_DIR / "deformed_images"

REFERENCE_PATH = INPUT_DIR / "reference_orange.png"
CURRENT_DEFORMED_PATH = INPUT_DIR / "deformed_orange.png"

OUTPUT_BATCH_DIR = PROJECT_DIR / "outputs" / "real_visual_batch"
REAL_INFERENCE_SAMPLES_DIR = PROJECT_DIR / "outputs" / "real_inference_samples"

RAW_PHOTO_DIR = PROJECT_DIR / "basler_fotos"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFORMED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_BATCH_DIR.mkdir(parents=True, exist_ok=True)
REAL_INFERENCE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
RAW_PHOTO_DIR.mkdir(parents=True, exist_ok=True)


ARUCO_IDS = [1, 2, 3]
ARUCO_TYPE = "DICT_4X4_100"

EXPECTED_N_TRIANGLES = None

LOWER_ORANGE = (0, 5, 85)
UPPER_ORANGE = (60, 210, 255)

RUN_PIPELINE_AFTER_CAPTURE = True


def next_deformed_filename() -> tuple[int, Path]:
    existing = sorted(DEFORMED_DIR.glob("deformed_orange_*.png"))

    ids = []
    for path in existing:
        try:
            ids.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass

    next_id = max(ids) + 1 if ids else 1
    path = DEFORMED_DIR / f"deformed_orange_{next_id:06d}.png"

    return next_id, path


def run_pipeline_for_deformed_image(sample_id: int, deformed_path: Path) -> None:
    if not REFERENCE_PATH.exists():
        print("Cannot run pipeline: reference_orange.png does not exist yet.")
        print("Press 'r' first to save the reference image.")
        return

    output_dir = OUTPUT_BATCH_DIR / deformed_path.stem

    print()
    print("=" * 80)
    print(f"Running pipeline for: {deformed_path.name}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    try:
        result = run_real_visual_pipeline(
            reference_image_path=REFERENCE_PATH,
            deformed_image_path=deformed_path,
            output_dir=output_dir,
            required_ids=ARUCO_IDS,
            aruco_type=ARUCO_TYPE,
            expected_n_triangles=EXPECTED_N_TRIANGLES,
            lower_orange=LOWER_ORANGE,
            upper_orange=UPPER_ORANGE,
        )

        sample_dir = write_real_inference_sample(
            sample_id=sample_id,
            real_pipeline_result=result,
            real_samples_dir=REAL_INFERENCE_SAMPLES_DIR,
        )

        print(f"Pipeline finished for: {deformed_path.name}")
        print(f"Inference sample saved to: {sample_dir}")

    except Exception as error:
        print(f"Pipeline failed for: {deformed_path.name}")
        print(error)
        print(traceback.format_exc())


camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
print(f"Connected to: {camera.GetDeviceInfo().GetModelName()}")

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

print()
print("Controls:")
print("  r   save/overwrite reference image as inputs/reference_orange.png")
print("  d   save new deformed image and run pipeline")
print("  ESC close camera")
print()

cv2.namedWindow("Basler Feed", cv2.WINDOW_NORMAL)

try:
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()

            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            cv2.imshow("Basler Feed", rotated_img)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key == ord("r"):
                cv2.imwrite(str(REFERENCE_PATH), rotated_img)
                print(f"Reference image saved/overwritten: {REFERENCE_PATH}")

            elif key == ord("d"):
                sample_id, deformed_path = next_deformed_filename()

                cv2.imwrite(str(deformed_path), rotated_img)

                shutil.copy2(deformed_path, CURRENT_DEFORMED_PATH)

                print(f"Deformed image saved: {deformed_path}")
                print(f"Current pipeline image updated: {CURRENT_DEFORMED_PATH}")

                if RUN_PIPELINE_AFTER_CAPTURE:
                    run_pipeline_for_deformed_image(
                        sample_id=sample_id,
                        deformed_path=deformed_path,
                    )
                elif key == ord("s"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    raw_path = RAW_PHOTO_DIR / f"foto_{timestamp}.jpg"
                    cv2.imwrite(str(raw_path), rotated_img)
                    print(f"Raw photo saved: {raw_path}")

        grabResult.Release()

finally:
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()

    for _ in range(4):
        cv2.waitKey(1)

    print("Camera safely released.")