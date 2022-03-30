FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# add source code
ADD . /opt/fastaugment

# compile
RUN cd /opt/fastaugment/pytorch &&\
    TORCH_CUDA_ARCH_LIST="7.0;7.2;7.5;8.0;8.6" python3 setup.py install

# try to import the module in Python
RUN python3 -c "import fast_augment_torch; print('Yay!')"
