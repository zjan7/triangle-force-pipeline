from src.pipeline import run_one_sample


if __name__ == "__main__":
    run_one_sample(
        sample_id=1,
        grid_width=0.1,
        grid_height=0.1,
        target_triangles=200,

    )