# In order to reproduce results, follow instructions of how to download and arrange datasets!

**Training Dataset**  
- **MLL23 [1]**: 41,621 peripheral blood single-cell images across 18 cell types.  
  https://github.com/marrlab/MLL23

```
cd dataset

curl -L -o mll23.zip "https://zenodo.org/api/records/14277609/draft/files-archive"

unzip mll23.zip -d mll23 && rm mll23.zip

cd mll23

for f in *.zip; do
    unzip "$f" && rm "$f"
done

find . -name "__MACOSX" -type d -exec rm -rf {} +
```
---

**Evaluation Datasets**  
- **Acevedo [2]**: 17,092 peripheral blood single-cell images labeled into 11 classes.  
  https://data.mendeley.com/datasets/snkd93bnjr/1  

```
cd dataset

curl -L -o acevedo.zip https://data.mendeley.com/public-files/datasets/snkd93bnjr/files/2fc38728-2ae7-4a62-a857-032af82334c3/file_downloaded

unzip acevedo.zip && rm acevedo.zip && mv PBC_dataset_normal_DIB acevedo

find . -name "__MACOSX" -type d -exec rm -rf {} +

conda activate cytosae
python arrange_dataset.py --data_name acevedo
```

---

- **BMC [4]**: 171,373 expert-annotated bone marrow smear cells. You can download manually from  
  https://doi.org/10.7937/TCIA.AXH3-T579 and place under `dataset` folder or on command-line:

```
cd dataset

curl -L -o bmc.zip "https://storage.googleapis.com/kaggle-data-sets/1855740/3082954/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250709%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250709T212420Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5678c0c3e835feafa903eecc7469cac86d33fc5f072bd94f28948678dd62ee0250fb6e02c307907ad6358590c5201d38558e8f19d925b4c418f608846f85c25ab5f6b2c792828fb47deb6236f0f5e432298445dfb00f2fdee4b38beff08fff956573fd1b32dad7dc6cb017fff8adf6a292f6d1836f8a4efdb08e635814b6581524fd158679feac73969d39b4ff06956fc1d83c335a257683e3abfdb60a059dd56a91b710d00071c58819e623ef0ad28bda78ca82f396fa83edbe3a2643038b6059eb31c6909d60253e8f3915b065af48bff515a7877dca46653e214cdd44f6d4de13ce1737769297f9dbb933c8ef6fcb04c0e7c5ecd9d4814228bdb3e8f98755"

unzip bmc.zip -d bmc && rm bmc.zip

find . -name "__MACOSX" -type d -exec rm -rf {} +

conda activate cytosae
python arrange_dataset.py --data_name bmc
```

---

- **Matek19 [3]**: 18,365 expert-labeled peripheral blood single-cell images, grouped into 15 classes. You can download manually from https://doi.org/10.7937/tcia.2019.36f5o9ld and place under `dataset` folder or on command-line:  
```
cd dataset

curl -L -o matek.zip "https://storage.googleapis.com/kaggle-data-sets/4260883/7384489/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250711%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250711T133746Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=97b79a32baab7327d6bdc65f6926d2c5f0c6aa325d50782b1094f6e25604ee7a7209dabd1dcb55a980b8034f61ad2d9d0c79216c783e79a003b55c2dd60dbf4f9e84a5f41df0d47e32c93c99c8437de4cb7baa1c3ec11f4388da89df6fa3dd16e941940cd6bab6491f872e685cf11b632e21511e91dc8bc124aac5e6daaf08941d65e274d9b9779f20dde3f6821ca8da6664fe2d7859fd09e8c3258ce89e734f6c52266452d69cee1e666631fb408696e55d91f7f9829eb87fe527ecba8f533cd53e07b7590de0167d1a33a852cdd9acbb6fd1a68899882ecc0d795419913d69f4046f15cbcc5a46ce84b88b56f946bf053004372995024f8351ea3b50fbad12"

unzip matek.zip -d matek && rm matek.zip

find . -name "__MACOSX" -type d -exec rm -rf {} +

conda activate cytosae
python arrange_dataset.py --data_name matek
```

---

- **AML_Hehr [5]**: Patient-level single-cell images from 189 subjects, including four genetic AML subtypes and controls.  
  https://doi.org/10.7937/6ppe-4020
-> Download from the official website manually and place under dataset folder

### Dataset Structure

To use `datasets.load_dataset`, arrange your dataset in a class-wise structure like (already done for mentioned datasets using [arrange_dataset.ipynb](arrange_dataset.ipynb)):

```
dataset_root/
├── class_1/
│   ├── image_1.ext
│   ├── image_2.ext
│   └── ...
├── class_2/
└── ...
```

Then, define the dataset path in `tasks/utils.py`. Example for class-wise structured data:

```python
DATASET_INFO = {
    "mll23": {
        "path": "dataset/mll23",
        "split": "train",
    },
}
```

If the dataset is not structured class-wise, you can export image paths and labels to a [CSV](../csv/bm_train_test.csv) and define the dataset path in `tasks/utils.py` as follows:

```python
"custom_data": {
    "path": "csv", 
    "data_files": "./csv/bm_train_test.csv",
    "split": "train",
},
```
Then use a custom dataloader:
```
dataset = load_dataset(**DATASET_INFO[dataset_name])
if dataset_name in ['custom_data']:
    def transform_function(example):
        # This function will be called on each example when it is accessed.
        try:
            example["image"] = [Image.open(img).convert("RGB") for img in example["image"]]
        except Exception as e:
            # Optionally handle errors, e.g., mark the example as invalid or return None.
            print(f"Error processing {example['image']}: {e}")
            example["image"] = [None]
        try:
            # print(example["label"])
            example["label"] = [int(label) for label in example["label"]]
        except Exception as e:
            print(f"Error processing {example['label']}: {e}")
            example["label"] = [None]
        return example 
    dataset.set_transform(transform_function)
```

---