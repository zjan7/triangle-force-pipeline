from src.pipeline_loop import run_dataset_generation


if __name__ == "__main__":
    run_dataset_generation(
        n_accepted_samples=50,
        max_attempts=52,
        grid_width=0.1,
        grid_height=0.1,
        target_triangles=200,
    )

