# Install climate-learn without editable option (-e)
```
pip uninstall climate-learn
cd /lustre/orion/lrn036/proj-shared/xf9/climate-learn
pip install . --prefix=/lustre/orion/lrn036/world-shared/xf9/torch26
```

# Install conda-pack
```
conda install conda-pack -c conda-forge
```

# Pack the conda environment
```
cd /lustre/orion/lrn036/world-shared/xf9/
conda pack -p /lustre/orion/lrn036/world-shared/xf9/torch26
```

