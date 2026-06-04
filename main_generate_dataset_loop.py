from src.pipeline_loop import run_dataset_generation


if __name__ == "__main__":
    run_dataset_generation(
    n_accepted_samples=1,
    max_attempts=1,
    target_triangles=200,
    aspect_ratio=1.0,
    packing_factor=2.5,
)

