# vr2p
Tools for analyzing Gimbl VR and suite2p imaging data

# Usage
To load in processed data:

```python
data = vr2p.ExperimentData(zarr_path)
```
Folder path can be both on disk and gs://

See tutorial notebooks for how to extract relevant data.

## Multiday registration
Combine with multiday-suite2p repo to perform multiday registration of same FOV across days.
Notebook at 'tutorials/Create multi-day vr2p ExperimentData object.ipynb' shows how to aggregate this data in to one vr2p object.


## Install
* Create anaconda environment from yml file:
    ```
    conda env create -f environment.yml
    ``` 
* Install module from within repo directory:
    ```
    pip install -e .
    ```