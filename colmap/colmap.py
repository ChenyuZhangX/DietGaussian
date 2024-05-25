import shutil
from pathlib import Path

import pycolmap
from pycolmap import logging


def run():
    name = 'toy'
    max_num_features = 2048 * 4 * 8
    output_path = Path(f"./data/{name}/")
    image_path = Path(f'./data/{name}/images') #output_path / "Fountain/images"
    database_path = output_path / "database.db"
    sfm_path = output_path / "sfm"
    undistort_path = output_path / "undistort"  

    output_path.mkdir(exist_ok=True)
    logging.set_log_destination(logging.INFO, output_path / "INFO.log.")  # + time

    if database_path.exists():
        database_path.unlink()

    pycolmap.extract_features(database_path, image_path, sift_options={"max_num_features": max_num_features})
    pycolmap.match_exhaustive(database_path)
    num_images = pycolmap.Database(database_path).num_images

    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)

    maps = pycolmap.incremental_mapping(database_path, image_path, sfm_path)
    # maps[0].write(output_path)
    # with enlighten.Manager() as manager:
    #     with manager.counter(total=num_images, desc="Images registered:") as pbar:
    #         pbar.update(0, force=True)
    #         recs = pycolmap.incremental_mapping(
    #             database_path,
    #             image_path,
    #             sfm_path,
    #             initial_image_pair_callback=lambda: pbar.update(2),
    #             next_image_callback=lambda: pbar.update(1),
    #         )
    # for idx, rec in recs.items():
    #     logging.info(f"#{idx} {rec.summary()}")
    
    pycolmap.undistort_images(undistort_path, sfm_path / "0", image_path)



if __name__ == "__main__":
    run()