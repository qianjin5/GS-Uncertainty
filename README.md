# 1. Installation
## Clone this repository.
```
git clone git@github.com:qianjin5/GS-Uncertainty.git
```

## Install dependencies.
1. create an environment
```
conda create -y -n 3dgs-env python=3.8
conda activate 3dgs-env
```

2. install pytorch and other dependencies.
```
pip install plyfile tqdm torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install cudatoolkit-dev=11.7 -c conda-forge
```

3. install submodules
```
CUDA_HOME=PATH/TO/CONDA/envs/3dgs-env/pkgs/cuda-toolkit/ pip install submodules/diff-gaussian-rasterization submodules/simple-knn/

# tetra-nerf for Marching Tetrahedra
cd submodules/tetra-triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
cmake .

# you can specify your own cuda path
# export CPATH=PATH/TO/CONDA/envs/3dgs-env/pkgs/cuda-toolkit/:$CPATH
make 
pip install -e .
```

4. install viewer (in new terminal, do not activate conda)
```
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev

cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
```

# 2. Preparation
Download nerf_synthetic and nerf_llff_data datasets.

# 3. Run training
python train.py --source_path PATH_TO_DATASET_FOLDER/nerf_synthetic/chair --iterations 5000 --config configs/my.json

# 4. Visualize
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output/EXP_ID/
