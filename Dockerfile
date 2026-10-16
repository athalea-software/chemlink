# 1. BASE Y VARIABLES DE ENTORNO (De tu archivo original)
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # Priorizamos el PATH de Conda para ACPYPE, pero mantenemos GROMACS y CUDA
    PATH=/opt/miniconda/envs/bio/bin:/opt/miniconda/bin:/usr/local/gromacs/bin:/usr/local/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    GMXRC=/usr/local/gromacs/bin/GMXRC

# 2. DEPENDENCIAS DEL SISTEMA (Unificado)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl wget git unzip \
    build-essential cmake autoconf automake libtool csh \
    libeigen3-dev libfftw3-dev libxml2-dev libx11-dev libxt-dev \
    libncurses5-dev libqhull-dev libnetcdf-dev openmpi-bin \
    libopenmpi-dev libfftw3-mpi-dev ocl-icd-libopencl1 \
    opencl-headers clinfo swig zlib1g-dev libcairo2-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. INSTALACIÓN DE MINICONDA (Para solucionar ACPYPE/AmberTools)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh

# 4. ENTORNO 'BIO' (Aquí vive ACPYPE, OpenBabel, Meeko y fpocket)
# OpenBabel se instala solo por el comando `obabel`, que ChemLink ejecuta como
# programa aparte; el código de ChemLink no importa OpenBabel.
# fpocket viene de bioconda (binario probado): evita compilar su qhull vendorizado,
# cuya versión 2015.2 falla con QH6047 al construirse con toolchains modernos.
RUN /opt/miniconda/bin/conda create -n bio -y --override-channels \
    -c conda-forge -c bioconda \
    python=3.10 \
    pip \
    ambertools \
    acpype \
    "openbabel>=3.2.1" \
    pdbfixer \
    openmm \
    rdkit \
    meeko \
    scipy \
    gemmi \
    matplotlib \
    fpocket \
    tqdm \
    pandas \
    && /opt/miniconda/bin/conda clean -afy

# 5. HERRAMIENTAS DE DOCKING (Compilaciones manuales de tu original)

# AutoGrid4 y AutoDock4 (De tu original)
RUN git clone https://github.com/ccsb-scripps/AutoGrid.git /tmp/autogrid && \
    cd /tmp/autogrid && autoreconf -i && ./configure --prefix=/usr/local && \
    make -j$(nproc) install && \
    if [ -f /usr/local/bin/autogrid ]; then mv /usr/local/bin/autogrid /usr/local/bin/autogrid4; fi && \
    rm -rf /tmp/autogrid 

RUN git config --global http.postBuffer 524288000 && \
    git clone --depth 1 https://github.com/ccsb-scripps/AutoDock4.git /tmp/autodock4 && \
    cd /tmp/autodock4 && autoreconf -i && ./configure --prefix=/usr/local && \
    make -j$(nproc) install && \
    rm -rf /tmp/autodock4

# AutoDock-GPU (Optimizado para arquitecturas sm_86, 89, 90 y 120)
ARG AUTODOCK_GPU_TARGETS="86 89 90 120"
RUN git clone https://github.com/ccsb-scripps/AutoDock-GPU.git /tmp/autodock-gpu && \
    cd /tmp/autodock-gpu && \
    sed -i 's/-std=c++11/-std=c++14/g' Makefile.Cuda && \
    make DEVICE=CUDA NUMWI=128 \
      TARGETS="${AUTODOCK_GPU_TARGETS}" \
      CUDA_PATH=/usr/local/cuda \
      CUDA_INC_PATH=/usr/local/cuda/include \
      CUDA_LIB_PATH=/usr/local/cuda/lib64 \
      GPU_INCLUDE_PATH=/usr/local/cuda/include \
      GPU_LIBRARY_PATH=/usr/local/cuda/lib64 \
      CPU_INCLUDE_PATH=/usr/include \
      CPU_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
      -j"$(nproc)" && \
    cp bin/autodock_gpu_128wi /usr/local/bin/autodock-gpu && \
    chmod +x /usr/local/bin/autodock-gpu && \
    rm -rf /tmp/autodock-gpu

# 6. GROMACS 2025.4 (Compilación con soporte para RTX 3050 y otras)
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6-linux-x86_64.sh -O /tmp/cmake-install.sh && \
    bash /tmp/cmake-install.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake-install.sh && \
    wget -q https://ftp.gromacs.org/gromacs/gromacs-2025.4.tar.gz -O /tmp/gromacs.tar.gz && \
    tar xzf /tmp/gromacs.tar.gz -C /tmp && cd /tmp/gromacs-2025.4 && mkdir build && cd build && \
    /usr/local/bin/cmake .. \
        -DGMX_BUILD_OWN_FFTW=ON \
        -DREGRESSIONTEST_DOWNLOAD=OFF \
        -DGMX_GPU=CUDA \
        -DGMX_MPI=ON \
        -DGMX_SIMD=AVX2_256 \
        -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DGMX_CUDA_TARGET_SM="75;80;86;89;90;100;120" \
        -DGMX_CUDA_TARGET_COMPUTE="90" && \
    make -j$(nproc) && make install && rm -rf /tmp/gromacs*

# 7. AutoDock Vina (prebuilt static binaries — no compile, no Boost)
ARG VINA_VERSION=1.2.5
RUN curl -fL -o /usr/local/bin/vina \
      "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v${VINA_VERSION}/vina_${VINA_VERSION}_linux_x86_64" && \
    curl -fL -o /usr/local/bin/vina_split \
      "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v${VINA_VERSION}/vina_split_${VINA_VERSION}_linux_x86_64" && \
    chmod +x /usr/local/bin/vina /usr/local/bin/vina_split && \
    vina --version

# 8. CONFIGURACIÓN FINAL
WORKDIR /app/chemlink
COPY . .

RUN /opt/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /opt/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /opt/miniconda/bin/conda env create -n gmx_mmpbsa -f config/environments/gmx_mmpbsa.yml \
    && /opt/miniconda/envs/gmx_mmpbsa/bin/python -m pip install --no-cache-dir --no-deps gmx-MMPBSA \
    && /opt/miniconda/bin/conda clean -afy

# Dependencias Python del proyecto
RUN /opt/miniconda/envs/bio/bin/python -m ensurepip && \
    /opt/miniconda/envs/bio/bin/python -m pip install --no-cache-dir \
      biopython numpy rich "prompt_toolkit>=3.0"

RUN echo "source /usr/local/gromacs/bin/GMXRC" >> /etc/bash.bashrc

RUN printf '#!/usr/bin/env bash\nexport PYTHONPATH=/app:/app/chemlink:${PYTHONPATH:-}\nset +u; [ -f /usr/local/gromacs/bin/GMXRC ] && source /usr/local/gromacs/bin/GMXRC; set -u\nexec /opt/miniconda/envs/bio/bin/python -m chemlink.cli.main "$@"\n' \
    > /usr/local/bin/chemlink \
    && chmod +x /usr/local/bin/chemlink

CMD ["/bin/bash"]
