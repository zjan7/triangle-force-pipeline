#Dit is het deel om Sweder zijn testopstelling detectie op te doen
from pathlib import Path
from src.real_visual_pipeline import run_real_visual_pipeline
from src.real_inference_writer import write_real_inference_sample

REFERENCE_IMAGE = Path("inputs/reference_orange.png") #hier kan je een path instellen naar enige locatie die je wilt, het makkelijkste is momenteel om in de powershell dit in te voeren; mkdir inputs, hiermee maak je een mapje inputs en dan in inputs moet je de afbeeldingen uploaden en verander dan de namen van de paths naar de namen van je afbeelding
DEFORMED_IMAGE = Path("inputs/deformed_orange.png")

OUTPUT_DIR = Path("outputs/real_visual_test")

ARUCO_IDS = [1, 2, 3]
ARUCO_TYPE = "DICT_4X4_100"
EXPECTED_N_TRIANGLES = None #dit boeit niet zoveel maar als we bijvoorbeeld 20 driehoeken vinden kan je dat hier zeggen

#hiermee heb je een oranje threshold, als er teveel noise is: verhoog S of V of verlaag H, als de driehoek niet gedetecteerd wordt: verlaag S en V of verhoog H.

LOWER_ORANGE = (0, 5, 85)
UPPER_ORANGE = (60, 210, 255)

SAMPLE_ID = 1
if __name__ == "__main__": #hiermee run je de pipeline, hier moet normaal niets aangepast in worden
    pipeline_result = run_real_visual_pipeline(
        reference_image_path=REFERENCE_IMAGE,
        deformed_image_path=DEFORMED_IMAGE,
        output_dir=OUTPUT_DIR,
        required_ids=ARUCO_IDS,
        aruco_type=ARUCO_TYPE,
        expected_n_triangles=EXPECTED_N_TRIANGLES,
        lower_orange=LOWER_ORANGE,
        upper_orange=UPPER_ORANGE,
    )

    write_real_inference_sample(
        sample_id=SAMPLE_ID,
        real_pipeline_result=pipeline_result,
        real_samples_dir=Path("outputs/real_inference_samples"),
    )