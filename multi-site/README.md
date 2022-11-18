The multi-site model to select a set of CpG sites that are associated with baseline eGFR/eGFR slope.

### run_whole_selection.py
Run the whole selection procedure using all the dataset.

To run the script, type:

```
python run_whole_selection.py [target variable] [data dir] [methylation file] [sample annotation file]
```

For a dataset containing ~450,000 CpG sites and ~1,200 samples, the estimated running time is ~10 hours if 40 CPU cores are used.

### final_model_with_selected.py
Build the final LASSO model on all the data using the selected set of CpG sites.

To run the script, type:

```
python final_model_with_selected.py [target variable] [data dir] [methylation file] [sample annotation file]
```

The estimated running time is ~1 minute if 40 CPU cores are used.

### output_whole_eGFR_CKDEPI
The final model for baseline eGFR.

### output_whole_eGFR_slopes
The final model for eGFR slope.
