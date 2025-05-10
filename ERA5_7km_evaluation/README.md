**ORBIT Downscaling Experiments Proof of Principle**

Modified from Climate Learn.

(1) Install your conda environment

(2) Then do `pip install -e .` to install the package to the environment

(3) Do  `pip install -r requirements.txt`

(4) Go to tutorial folder. Downscaling.py is the downscaling super resolution proof of principle code

**Run Script for Downscaling and Visualization**


To run it, use the job scheduler to run the job launch_downscaling.sh

`python ./downscaling.py --max_epochs 30 /lustre/orion/lrn036/world-shared/ERA5_npz/5.625_deg/ /lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg/ vit t2m`

In the above line --max_epochs is the maximum number of epochs. "/lustre/orion/lrn036/world-shared/ERA5_npz/5.625_deg/" is the input coarse resolution data path. "/lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg/" is the ground truth high resolution data path. 
"vit" is an AI architecture of choice. Besides "**vit**", you can also use "**resnet**", "**unet**" or "**res_slimvit**" architecture by setting the corresponding flag.
"t2m" is the output variable. Some other example output variables to choose include t2m (surface temperature), z500 (500 hpa geopotential), t850 (temperature at 850 hpa), and u10 (u component of wind at 10m). 
The input variables can be changed in the code downscaling.py
You can also request multiple nodes by changing `#SBATCH --nodes=1`  and set --nodes to be larger than 1.


(5) To visualize the input and output, run launch_visualize.sh afterwards. Use only a single node with a single GPU. In visualize.py, do not forget to change the AI architecture choice and checkpoint path according to the training setup.

(6) Available training losses include MSE, MAE, latitude weighted MSE, Pearson Score, Anomaly Coefficient. Most recently, hybrid perceptual loss is implemented.  Training loss can be changed in ./src/climate-learn/util/loader.py


**Available Training Data**  
ERA5 5.6 degree "/lustre/orion/lrn036/world-shared/ERA5_npz/5.625_deg/"   
ERA5 1.4 degree "/lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg/"  
ERA5 1.0 degree "/lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/ERA5-1hr-superres/1.0_deg/"  
ERA5 0.25 degree "/lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/ERA5-1hr-superres/0.25_deg/"   
PRISM 16 km "/lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/prism-superres/10.0_arcmin/"  
PRISM 4 km "/lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/prism-superres/2.5_arcmin/"  

