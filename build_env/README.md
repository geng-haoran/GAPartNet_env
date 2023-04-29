build env for RL-Pose

cuda 11.3

nvcc 11.3

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install torchdata==0.3.0

isaacgym 4

pointnet_ops

Open3d
- cmake command:
    ```
    cmake \
    -DPython3_ROOT="/scratch/genghaoran/anaconda3/envs/pg38/bin/" \
    -DWITH_OPENMP=ON \
    -DWITH_SIMD=ON \
    -DBUILD_PYTORCH_OPS=ON \
    -DBUILD_CUDA_MODULE=ON \
    -DBUILD_COMMON_CUDA_ARCHS=ON \
    -DGLIBCXX_USE_CXX11_ABI=OFF \
    -DBUILD_JUPYTER_EXTENSION=ON \
    -DCMAKE_CUDA_COMPILER=$(which nvcc) \
    -DCMAKE_INSTALL_PREFIX="/scratch/genghaoran/build_env//build_env/open3d_install" \
    ..                         
    ```
    ```
    make -j$(nproc)

    make install

    # Activate the virtualenv first
    # Install pip package in the current python environment
    make install-pip-package

    # Create Python package in build/lib
    make python-package

    # Create pip wheel in build/lib
    # This creates a .whl file that you can install manually.
    make pip-package
    ```