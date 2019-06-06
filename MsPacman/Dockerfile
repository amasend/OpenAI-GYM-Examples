FROM ubuntu
RUN apt-get update && apt-get install -y python3 \
python3-pip \
git \
cmake \
#mesa-common-dev \
vim \
libz-dev 
#libglu1-mesa-dev \
#freeglut3-dev \ 
RUN pip3 install gym \ 
numpy \
keras \
tensorflow \
matplotlib \
pandas 
RUN pip3 install gym[atari]
RUN git clone https://github.com/amasend/OpenAI-GYM-Examples.git