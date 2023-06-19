## Configuration
Project configuration files are located in `config` folder: `config/<input_type>.toml`.
There are four supported input types: `smpl, smplh, rawpc, hands`. 
All the following commands are described for `smplh`, update other configs similarly for corresponding input type. 

## SMPL, SMPL+H and MANO model
The project uses [smplx](https://github.com/vchoutas/smplx) package to work with SMPL body model.
Follow [this](https://github.com/vchoutas/smplx#downloading-the-model) link for the instruction on setting it up.
After installing smplx make sure to set `SMPLX_PATH=<path to folder with SMPL, SMPL+H, and MANO models>` in 
`config/smplh.toml` to point to the folder with the [following](https://github.com/vchoutas/smplx#model-loading) 
structure.


## Downloading data
### GRAB
Instructions on downloading and setting up the GRAB dataset.

1. Register and download data from the official [website](https://grab.is.tue.mpg.de/index.html). 
   Required data: Subject 1 - Subject 10 (GRAB parameters), Subject Shape Templates, and GRAB objects.
2. Extract data using official [extracting code](https://github.com/otaheri/GRAB/#getting-started), 
   the expected folder structure is provided below (`data_smplh` is generated during the next step). 
3. Convert SMPL-X parameters to SMPL+H (instructions are [here](../popup/utils/smplx2smplh/README.md)).

#### Folders structure
```bash
<extracted GRAB path>
├── grab
│   ├── s1
│   │   ├── airplane_fly_1.npz
│   │   ├── ...
│   ├── ...
│   ├── s10
├── tools
│   ├── object_meshes
│   │   ├── contact_meshes
│   │   │   ├── airplane.ply
│   │   │   ├── ...
│   ├── subject_meshes
│   ├── ...
├── data_smplh
│   ├── s1
│   │   ├── banana_eat_1
│   │   │   ├── sequence_data.pkl
│   │   ├── ...
│   ├── ...
│   ├── s10
```

### BEHAVE
Instructions on downloading and setting up the BEHAVE dataset.

1. Download data from the official [website](https://virtualhumans.mpi-inf.mpg.de/behave/license.html). 
   Required data: Date 01 - Date 07 sequences, Scanned objects, and Annotations at 30fps.
2. Extract data, following the folder structure below.

#### Folders structure
```bash
<extracted BEHAVE path>
├── behave-30fps-params-v1
│   ├── sequences
│   │   ├── Date01_Sub01_backpack_back
│   │   ├── ...
├── objects
│   ├── backpack
│   ├── ...
├── sequences
│   ├── Date01_Sub01_backpack_back
│   ├── ...
```


## Preprocessing data

To preprocess the data follow the steps below:
1. After completing **all** the steps above, run:
  ```bash
  python preprocess_data.py grab behave -t smplh -r -G <extracted GRAB path> -B <extracted BEHAVE path> -T ./data/preprocessed/
  ```
  Notes:
  * `-t` controls input type, `smpl rawpc hands` can be specified in addition to `smplh`. To preprocess other input 
    types make sure to update corresponding configs in `config/<input_type>.toml`.
  * `-r` allows to use keypoints that were utilized for the model training (stored in the `./assets` folder).   

2. Optionally, change `GRAB_PATH` in `config/smplh.toml` (default: `./data/preprocessed/grab_smplh/`).
3. Optionally, change `BEHAVE_PATH` in `config/smplh.toml` (default: `./data/preprocessed/behave_smplh/`).
