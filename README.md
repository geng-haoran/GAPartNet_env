# GAPartNet_env
GAPartNet conda environment

## An worked environment:

build env for RL-Pose

### Step 1: CUDA and basic conda environment

python == 3.8

cuda, nvcc >= 11.3

torch >= 1.11: 
e.g. pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

### Step 2: cmake and open3d
- Notice: here you may need su/sudo
cmake >= 3.20: following the instruction [here](https://cmake.org/install/)

Some possible problems here:
1. Could not find OpenSSL
```
sudo aptitude install libssl-dev
```

2. Could NOT find X11 (missing: X11_X11_INCLUDE_PATH X11_X11_LIB)
```
sudo aptitude install libx11-dev
```

3. The RandR headers were not found
```
sudo apt-get install libxrandr-dev
```

4. The Xinerama headers were not found
```
sudo apt-get install libxinerama-dev
sudo apt-get install libsdl2-dev
```

5. The Xcursor headers were not found
```
sudo apt-get install libxcursor-dev
```

6. Cannot find matching libc++ and libc++abi libraries with version >=7
```
sudo aptitude install -y clang libc++-dev libc++abi-dev cmake ninja-build
```

open3d with pytorch extension (need to build from source)
install repo
```
git clone https://github.com/isl-org/Open3D
```
install dependences
```
util/install_deps_ubuntu.sh
```
build:
```
mkdir build
cd build
```

cmake command (remember to replace YOUR_PATH_TO_INSTALL_FOLDER & YOUR_PATH_TO_ANACONDA_OR_MINICONDA & YOUR_CONDA_ENV_NAME as your real paths or names):
```
cmake \
-DPython3_ROOT="/YOUR_PATH_TO_ANACONDA_OR_MINICONDA/envs/YOUR_CONDA_ENV_NAME/bin/" \
-DWITH_OPENMP=ON \
-DWITH_SIMD=ON \
-DBUILD_PYTORCH_OPS=ON \
-DBUILD_CUDA_MODULE=ON \
-DBUILD_COMMON_CUDA_ARCHS=ON \
-DGLIBCXX_USE_CXX11_ABI=OFF \
-DBUILD_JUPYTER_EXTENSION=ON \
-DCMAKE_CUDA_COMPILER=$(which nvcc) \
-DCMAKE_INSTALL_PREFIX="YOUR_PATH_TO_INSTALL_FOLDER/open3d_install" \
..                         
```

An example of tested cmake command:
```
cmake \
-DPython3_ROOT="~/miniconda3/envs/rlgpu/bin" \
-DWITH_OPENMP=ON \
-DWITH_SIMD=ON \
-DBUILD_PYTORCH_OPS=ON \
-DBUILD_CUDA_MODULE=ON \
-DBUILD_COMMON_CUDA_ARCHS=ON \
-DGLIBCXX_USE_CXX11_ABI=OFF \
-DBUILD_JUPYTER_EXTENSION=ON \
-DCMAKE_CUDA_COMPILER=$(which nvcc) \
-DCMAKE_INSTALL_PREFIX="/mnt/data/GAPartNet_env/open3d_install" \
..
```

```
make -j$(nproc)
```

```
make install
```

```
# Activate the virtualenv first
# Install pip package in the current python environment (recommended)
pip install jupyter_packaging jupyterlab
make install-pip-package

# Create Python package in build/lib
make python-package

# Create pip wheel in build/lib
# This creates a .whl file that you can install manually.
make pip-package
```

### Step 3: install other packages


#### pointnet_ops

https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib

clone repo
```
git clone git@github.com:erikwijmans/Pointnet2_PyTorch.git
```

install this repo
```
cd pointnet2_ops_lib
python setup.py install
```
#### epic_ops

https://github.com/geng-haoran/epic_ops

clone repo
```
git clone git@github.com:geng-haoran/epic_ops.git
```

install this repo
```
cd epic_ops
python setup.py develop
```

#### spconv
```
spconv: pip install spconv-cuxxx (https://github.com/traveller59/spconv)
```

#### pip
```
pip install wandb tensorboard ipdb gym tqdm rich opencv_python pytorch3d pyparsing pytorch_lightning addict yapf h5py sorcery  pynvml torchdata==0.5.1 einops
```
