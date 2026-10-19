# ChemLink — GPU image (GROMACS, AutoDock-GPU, Vina, fpocket, ACPYPE, gmx_MMPBSA).
#
# Two stages: everything is built in the CUDA "devel" image and the final image
# starts from the "runtime" one, which only ships the CUDA libraries NVIDIA
# allows to be redistributed. The final image keeps the licenses, the source
# code of what is built here and the exact package list in
# /usr/share/chemlink/third-party (see THIRD_PARTY_LICENSES.md).

# ─── Stage 1: build ───────────────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ARG THIRD_PARTY=/usr/share/chemlink/third-party

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git csh \
    build-essential autoconf automake libtool \
    libopenmpi-dev openmpi-bin zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p ${THIRD_PARTY}/src ${THIRD_PARTY}/licenses

# Miniforge (BSD-3-Clause, conda-forge only). Installed in /opt/miniconda
# because that is where ChemLink and its scripts look for the gmx_mmpbsa env.
ARG MINIFORGE_VERSION=26.7.2-0
ARG MINIFORGE_SHA256=281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05
RUN curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum -c - \
    && bash /tmp/miniforge.sh -b -p /opt/miniconda \
    && rm /tmp/miniforge.sh

# 'bio' environment. OpenBabel is only installed for the `obabel` command,
# which ChemLink runs as a separate program; ChemLink's code does not import
# OpenBabel. fpocket comes from bioconda (tested binary): this avoids building
# its vendored qhull, whose 2015.2 version fails with QH6047 on modern toolchains.
RUN /opt/miniconda/bin/conda create -n bio -y --override-channels \
    -c conda-forge -c bioconda \
    python=3.10 pip \
    ambertools acpype "openbabel>=3.2.1" pdbfixer openmm rdkit meeko scipy gemmi \
    matplotlib networkx fpocket tqdm pandas biopython numpy rich "prompt-toolkit>=3.0"

# Peptide docking: LightDock (GPL-3.0) and PRODIGY (Apache-2.0) come from PyPI;
# ChemLink runs both as separate programs.
RUN /opt/miniconda/envs/bio/bin/python -m pip install --no-cache-dir "lightdock>=0.9" "prodigy-prot>=2.4" \
    && /opt/miniconda/envs/bio/bin/lightdock3_setup.py --help > /dev/null \
    && /opt/miniconda/envs/bio/bin/prodigy --help > /dev/null

# Isolated gmx_MMPBSA environment (Python < 3.12, NumPy < 2).
COPY config/environments/gmx_mmpbsa.yml /tmp/gmx_mmpbsa.yml
RUN /opt/miniconda/bin/conda env create -n gmx_mmpbsa -f /tmp/gmx_mmpbsa.yml \
    && /opt/miniconda/envs/gmx_mmpbsa/bin/python -m pip install --no-cache-dir --no-deps gmx-MMPBSA

# Licenses and conda/pip package list, collected before `conda clean` deletes
# pkgs/*/info/licenses.
COPY scripts/collect_third_party_notices.py /tmp/collect_third_party_notices.py
RUN /opt/miniconda/bin/python /tmp/collect_third_party_notices.py \
      --conda-root /opt/miniconda --out ${THIRD_PARTY} \
    && /opt/miniconda/bin/conda clean -afy

# AutoGrid4 and AutoDock4 (GPL-2.0), pinned to a commit; their source code
# stays in the image.
ARG AUTOGRID_COMMIT=6d2847beaeac8ff43ca99094707fd74e3ca1ff37
ARG AUTODOCK4_COMMIT=192ecda05d7c566161046f0a1d604f3336e0cf3a
RUN git clone https://github.com/ccsb-scripps/AutoGrid.git /tmp/autogrid \
    && cd /tmp/autogrid && git checkout --quiet ${AUTOGRID_COMMIT} \
    && git archive --format=tar.gz --prefix=autogrid-${AUTOGRID_COMMIT}/ \
         -o ${THIRD_PARTY}/src/autogrid-${AUTOGRID_COMMIT}.tar.gz HEAD \
    && cp COPYING ${THIRD_PARTY}/licenses/AutoGrid-COPYING \
    && autoreconf -i && ./configure --prefix=/usr/local && make -j"$(nproc)" install \
    && if [ -f /usr/local/bin/autogrid ]; then mv /usr/local/bin/autogrid /usr/local/bin/autogrid4; fi \
    && rm -rf /tmp/autogrid
RUN git config --global http.postBuffer 524288000 \
    && git clone https://github.com/ccsb-scripps/AutoDock4.git /tmp/autodock4 \
    && cd /tmp/autodock4 && git checkout --quiet ${AUTODOCK4_COMMIT} \
    && git archive --format=tar.gz --prefix=autodock4-${AUTODOCK4_COMMIT}/ \
         -o ${THIRD_PARTY}/src/autodock4-${AUTODOCK4_COMMIT}.tar.gz HEAD \
    && cp COPYING ${THIRD_PARTY}/licenses/AutoDock4-COPYING \
    && autoreconf -i && ./configure --prefix=/usr/local && make -j"$(nproc)" install \
    && rm -rf /tmp/autodock4

# AutoDock-GPU (GPL-2.0), optimized for sm_86, 89, 90 and 120.
ARG AUTODOCK_GPU_COMMIT=be06a135181178c8631653e1e32d9ece86b5b4d8
ARG AUTODOCK_GPU_TARGETS="86 89 90 120"
RUN git clone https://github.com/ccsb-scripps/AutoDock-GPU.git /tmp/autodock-gpu \
    && cd /tmp/autodock-gpu && git checkout --quiet ${AUTODOCK_GPU_COMMIT} \
    && git archive --format=tar.gz --prefix=autodock-gpu-${AUTODOCK_GPU_COMMIT}/ \
         -o ${THIRD_PARTY}/src/autodock-gpu-${AUTODOCK_GPU_COMMIT}.tar.gz HEAD \
    && cp LICENSE ${THIRD_PARTY}/licenses/AutoDock-GPU-LICENSE \
    && sed -i 's/-std=c++11/-std=c++14/g' Makefile.Cuda \
    && make DEVICE=CUDA NUMWI=128 \
      TARGETS="${AUTODOCK_GPU_TARGETS}" \
      CUDA_PATH=/usr/local/cuda \
      CUDA_INC_PATH=/usr/local/cuda/include \
      CUDA_LIB_PATH=/usr/local/cuda/lib64 \
      GPU_INCLUDE_PATH=/usr/local/cuda/include \
      GPU_LIBRARY_PATH=/usr/local/cuda/lib64 \
      CPU_INCLUDE_PATH=/usr/include \
      CPU_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
      -j"$(nproc)" \
    && cp bin/autodock_gpu_128wi /usr/local/bin/autodock-gpu \
    && chmod +x /usr/local/bin/autodock-gpu \
    && rm -rf /tmp/autodock-gpu

# GROMACS 2025.4 (LGPL-2.1). CMake (BSD-3-Clause) is only used in this stage.
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6-linux-x86_64.sh -O /tmp/cmake-install.sh \
    && mkdir -p /opt/cmake \
    && bash /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake \
    && rm /tmp/cmake-install.sh \
    && wget -q https://ftp.gromacs.org/gromacs/gromacs-2025.4.tar.gz -O ${THIRD_PARTY}/src/gromacs-2025.4.tar.gz \
    && tar xzf ${THIRD_PARTY}/src/gromacs-2025.4.tar.gz -C /tmp \
    && cp /tmp/gromacs-2025.4/COPYING ${THIRD_PARTY}/licenses/GROMACS-COPYING \
    && cd /tmp/gromacs-2025.4 && mkdir build && cd build \
    && /opt/cmake/bin/cmake .. \
        -DGMX_BUILD_OWN_FFTW=ON \
        -DREGRESSIONTEST_DOWNLOAD=OFF \
        -DGMX_GPU=CUDA \
        -DGMX_MPI=ON \
        -DGMX_SIMD=AVX2_256 \
        -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DGMX_CUDA_TARGET_SM="75;80;86;89;90;100;120" \
        -DGMX_CUDA_TARGET_COMPUTE="90" \
    && make -j"$(nproc)" && make install && rm -rf /tmp/gromacs*

# AutoDock Vina 1.2.5 (Apache-2.0): official static binaries.
ARG VINA_VERSION=1.2.5
RUN curl -fL -o /usr/local/bin/vina \
      "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v${VINA_VERSION}/vina_${VINA_VERSION}_linux_x86_64" \
    && curl -fL -o /usr/local/bin/vina_split \
      "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v${VINA_VERSION}/vina_split_${VINA_VERSION}_linux_x86_64" \
    && curl -fL -o ${THIRD_PARTY}/licenses/AutoDock-Vina-LICENSE \
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/v${VINA_VERSION}/LICENSE" \
    && chmod +x /usr/local/bin/vina /usr/local/bin/vina_split \
    && vina --version

# CHARMM36m: not redistributed in the repository; downloaded from the official
# source and verified. It lives outside /app/chemlink so that docker-compose,
# which mounts the repository there, does not hide it. For an image that will
# be handed to third parties, build with --build-arg WITH_CHARMM36=false (see
# THIRD_PARTY_LICENSES.md); each user then downloads it into /opt/forcefields.
ARG WITH_CHARMM36=true
ARG CHARMM36_SHA256=9f31db8502852bc8fcf36497c27db14f1b6509b33701e7cafc2187840b7d2f09
RUN mkdir -p /opt/forcefields \
    && if [ "$WITH_CHARMM36" = "true" ]; then \
         curl -fL --retry 3 -o /tmp/charmm36.tgz \
           "https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files%2Fcharmm36-jul2022.ff.tgz" \
         && echo "${CHARMM36_SHA256}  /tmp/charmm36.tgz" | sha256sum -c - \
         && tar -xzf /tmp/charmm36.tgz -C /opt/forcefields \
         && test -s /opt/forcefields/charmm36-jul2022.ff/forcefield.itp \
         && rm /tmp/charmm36.tgz; \
       fi

# ─── Stage 2: final image ─────────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH=/opt/miniconda/envs/bio/bin:/opt/miniconda/bin:/usr/local/gromacs/bin:/usr/local/bin:${PATH} \
    GMXRC=/usr/local/gromacs/bin/GMXRC \
    CHEMLINK_FORCEFIELD_DIR=/opt/forcefields
ARG THIRD_PARTY=/usr/share/chemlink/third-party

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git csh libgomp1 openmpi-bin libopenmpi3t64 zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/miniconda /opt/miniconda
COPY --from=builder /usr/local/gromacs /usr/local/gromacs
COPY --from=builder /usr/local/bin/autogrid4 /usr/local/bin/autodock4 /usr/local/bin/autodock-gpu \
                    /usr/local/bin/vina /usr/local/bin/vina_split /usr/local/bin/
COPY --from=builder /opt/forcefields /opt/forcefields
COPY --from=builder ${THIRD_PARTY} ${THIRD_PARTY}

# Fail here, rather than on the first run, if a binary is missing a library in
# the runtime image (libcuda is provided by the driver at run time).
RUN missing="$(for bin in /usr/local/bin/autogrid4 /usr/local/bin/autodock4 /usr/local/bin/autodock-gpu \
                          /usr/local/gromacs/bin/gmx_mpi; do \
                 ldd "$bin" | grep 'not found' | grep -v -e 'libcuda\.so' -e 'libnvidia' | sed "s|^|$bin: |"; \
               done)"; \
    if [ -n "$missing" ]; then echo "$missing"; exit 1; fi

WORKDIR /app/chemlink
COPY . .

# Third-party notices and this image's operating system packages.
RUN cp THIRD_PARTY_LICENSES.md LICENSE LICENSING.md ${THIRD_PARTY}/ \
    && dpkg-query -W -f='${Package}\t${Version}\t${source:Package}\t${source:Version}\n' \
       > ${THIRD_PARTY}/manifests/debian-packages.tsv

RUN echo "source /usr/local/gromacs/bin/GMXRC" >> /etc/bash.bashrc

RUN printf '#!/usr/bin/env bash\nexport PYTHONPATH=/app:/app/chemlink:${PYTHONPATH:-}\nset +u; [ -f /usr/local/gromacs/bin/GMXRC ] && source /usr/local/gromacs/bin/GMXRC; set -u\nexec /opt/miniconda/envs/bio/bin/python -m chemlink.cli.main "$@"\n' \
    > /usr/local/bin/chemlink \
    && chmod +x /usr/local/bin/chemlink

CMD ["/bin/bash"]
