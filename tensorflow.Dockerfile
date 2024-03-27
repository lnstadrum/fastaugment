FROM tensorflow/tensorflow:latest-gpu

# add source code
ADD . /opt/fastaugment

# install cmake
RUN apt update && apt install -y cmake

# compile
RUN cd /opt/fastaugment/tensorflow &&\
    cmake -B build && make -C build

# update PYTHONPATH
ENV PYTHONPATH=$PYTHONPATH:/opt/fastaugment/tensorflow

# try to import the module in Python
RUN python3 -c "import fast_augment; print('Yay!')"
