FROM tensorflow/tensorflow:latest-gpu

# add source code
ADD . /opt/fastaugment

# install cmake and gcc
RUN apt update && apt install -y cmake

# compile
RUN cd /opt/fastaugment &&\
    rm -rf build && mkdir -p build && cd build &&\
    cmake .. && make

# update PYTHONPATH
ENV PYTHONPATH=$PYTHONPATH:/opt/fastaugment/tensorflow

# try to import the module in Python
RUN python3 -c "import fast_augment; print('Yay!')"
