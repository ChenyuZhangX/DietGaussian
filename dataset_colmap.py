import shutil
import urllib.request
import zipfile
from pathlib import Path
import enlighten

import pycolmap
from pycolmap import logging


def run(name):
    # max_num_features = 1024 * 8
    max_num_features = 'ori'
    camera_model = 'SIMPLE_RADIAL'
    output_path = Path(f"train/{name}_{str(max_num_features)}_{camera_model}/")
    image_path = Path(f'/home/sml/3dgs/splat/tsinghua/{name}') #output_path / "Fountain/images"
    database_path = output_path / "database.db"
    sfm_path = output_path / "sfm"
    undistort_path = output_path / "undistort"

    output_path.mkdir(exist_ok=True)
    logging.set_log_destination(logging.INFO, output_path / "INFO.log.")  # + time

    if database_path.exists():
        database_path.unlink()

    # pycolmap.extract_features(database_path, image_path, camera_model=camera_model, sift_options={"max_num_features": max_num_features})
    pycolmap.extract_features(database_path, image_path, camera_model=camera_model)
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
    # run('flower_yard')
    # run('stele')
    run('ultraman_mask_assemble')