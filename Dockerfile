FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update                                          &&\
    apt-get install -y git-core vim binutils                &&\
    apt-get install -y cmake g++ gcc                        

RUN apt-get install -y flatpak            &&\
    apt-get clean

RUN flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo

RUN flatpak install -y flathub org.cloudcompare.CloudCompare

# Add our special entrypoint
#
RUN git clone -b stable https://github.com/chbrandt/docker_entrypoint.git && \
    ln -sf docker_entrypoint/entrypoint.sh /.
#
ENTRYPOINT ["/entrypoint.sh"]

ENV EXECAPP "CloudCompare"

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN python3 --version
RUN pip3 install laspy
