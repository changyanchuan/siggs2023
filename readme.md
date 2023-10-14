Code implementation for the lake detection contest in GISCUP 2023. 

The submitted gpkg file for evaluation is located at `./GPKG/lake_polygons_test.gpkg`.

## Requirements
- Ubuntu 20.04 LTS with Python 3.7.7
- `pip install -r requirements.txt`
- Suggested running environment: a virtual machine with an 8-core Intel Xeon CPU, 128GB RAM and an NVIDIA Tesla A100 GPU (40GB VRAM).

## Quick Start

1. Install the required packages by following the above `Requirements' section.
2. Copy all datasets ([download here](https://drive.google.com/drive/folders/1p5N7QQwNkC5is89_IfdQfOZ__dQia91x)) to `./data/`.
3. \[Option 1\] Train the model from the origin, and evaluate it on the test dataset. The output gpkg for test regions is dumped to `./data/lake_polygons_test1.gpkg`.
```bash
python main.py --dumpfile_uniqueid 1
```
3. \[Option 2\] Load the checkpoint  of the pre-trained model and evaluate it on the test dataset, where the checkpoint file can be download from [here](https://drive.google.com/drive/folders/1Wpr4aHaOiEE6Z8HAJQ4xbRWKJMtXSKA5). Move the downloaded pt file to `./exp/snapshot/`. The output gpkg for test regions is dumped to `./data/lake_polygons_test2.gpkg`.
```bash
python main.py --load_checkpoint --dumpfile_uniqueid 2
```

## Contact
Email changyanchuan@gmail.com if you have any queries.


## License
The source code for the site is under the MIT license, which you can find in the LICENSE.txt file.