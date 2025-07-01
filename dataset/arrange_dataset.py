# arrange acevedo, bmc and matek

from pathlib import Path
import shutil
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', required=True, choices=['acevedo', 'bmc', 'matek'])
args = parser.parse_args() 

# arrange acevedo
if args.data_name == 'acevedo':
    acevedo_path = Path('acevedo')
    img_files = acevedo_path.rglob('*.jpg')

    for img_file in img_files:
        cell_type = img_file.stem.split('_')[0]

        if cell_type == "MMY":
            new_dir = acevedo_path / "metamyelocyte"
        elif cell_type == "MY":
            new_dir = acevedo_path / "myelocyte"
        elif cell_type == "PMY":
            new_dir = acevedo_path / "promyelocyte"
        elif cell_type == "BNE":
            new_dir = acevedo_path / "neutrophil_band"
        elif cell_type == "SNE":
            new_dir = acevedo_path / "neutrophil_segmented"
        else:
            continue  # Skip other cell types

        new_dir.mkdir(exist_ok=True, parents=True)
        shutil.move(str(img_file), str(new_dir / img_file.name))

    print("Acevedo dataset is ready!!")

    # Arrange bmc dataset
elif args.data_name == 'bmc':
    old_root = Path('bmc/bone_marrow_cell_dataset')
    root = old_root.parent

    # Move images from subfolders to class-level folders
    for class_dir in tqdm(old_root.iterdir(),desc="Processing class folders"):
        if class_dir.is_dir():
            target_dir = root / class_dir.name
            target_dir.mkdir(exist_ok=True)

            for img_path in class_dir.rglob("*.jpg"):
                target_path = target_dir / img_path.name
                if target_path.exists():
                    print(f"Skipping duplicate: {target_path}")
                else:
                    shutil.move(str(img_path), str(target_path))

    shutil.rmtree(old_root)
    print(f"Removed: {old_root}")

    # Remove known broken image
    file_to_remove = root / "MYB/MYB_05527.jpg"
    if file_to_remove.exists():
        os.remove(file_to_remove)
        print(f"Removed: {file_to_remove}")
    else:
        print(f"File not found: {file_to_remove}")

    print("BMC dataset is ready!")

elif args.data_name == 'matek':
    matek_root = Path('matek')
    deep_data_path = matek_root / 'data' / 'data'

    # Move class folders (excluding 'augmented') directly under 'matek'
    for subfolder in deep_data_path.iterdir():
        if subfolder.name == 'augmented':
            print(f"Removing folder: {subfolder}")
            shutil.rmtree(subfolder)
        elif subfolder.is_dir():
            target_dir = matek_root / subfolder.name
            if target_dir.exists():
                print(f"Skipping existing folder: {target_dir}")
            else:
                print(f"Moving {subfolder} to {target_dir}")
                shutil.move(str(subfolder), str(target_dir))

    # Remove empty intermediate folders if needed
    if deep_data_path.exists() and not any(deep_data_path.iterdir()):
        deep_data_path.rmdir()
        print(f"Removed empty folder: {deep_data_path}")

    data_dir = matek_root / 'data'
    if data_dir.exists() and not any(data_dir.iterdir()):
        data_dir.rmdir()
        print(f"Removed empty folder: {data_dir}")

    # Remove all .dat files in matek root
    for dat_file in matek_root.glob("*.dat"):
        try:
            dat_file.unlink()
            print(f"Removed .dat file: {dat_file}")
        except Exception as e:
            print(f"Failed to remove {dat_file}: {e}")

    print("Matek dataset is ready!")

