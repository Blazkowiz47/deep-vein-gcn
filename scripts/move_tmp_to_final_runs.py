import os
from pathlib import Path
from tqdm import tqdm


def move_resnet_from_tmp_to_final_runs():
    print(list(Path("./tmp").glob("resent*")))
    print(list(Path("./tmp").iterdir()))
    for dir in tqdm(Path("./tmp").iterdir()):
        run_name = dir.stem
        dir = str(dir)
        if not run_name.startswith('resnet') or "intra" in dir:
            continue
        print(run_name)
        os.makedirs(f"./final_runs/{run_name}/")
        os.system(
            f"cp ./tmp/{run_name}/checkpoints/best_model.pt ./final_runs/{run_name}/best_model.pt"
        )


if __name__ == "__main__":
    move_resnet_from_tmp_to_final_runs()
