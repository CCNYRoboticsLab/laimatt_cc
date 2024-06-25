#!/bin/bash

cmake -DCMAKE_BUILD_TYPE=Release\
  -DEIGEN_ROOT_DIR=/usr/include/eigen3\
  -DCOMPILE_CC_CORE_LIB_WITH_TBB=ON\
  -DOPTION_USE_DXF_LIB=ON\
  -DOPTION_USE_SHAPE_LIB=ON\
  -DPLUGIN_IO_QLAS=ON\
  -DINSTALL_EXAMPLE_PLUGIN=ON\
  -DINSTALL_EXAMPLE_GL_PLUGIN=ON\
  -DINSTALL_EXAMPLE_IO_PLUGIN=ON\
  -DINSTALL_QADDITIONAL_IO_PLUGIN=ON\
  -DINSTALL_QANIMATION_PLUGIN=ON\
  -DINSTALL_QBROOM_PLUGIN=ON\
  -DINSTALL_QCOMPASS_PLUGIN=ON\
  -DINSTALL_QCANUPO_PLUGIN=ON\
  -DDLIB_ROOT=/usr/include\
  -DINSTALL_QCSF_PLUGIN=ON\
  -DINSTALL_QEDL_PLUGIN=ON\
  -DINSTALL_QFACETS_PLUGIN=ON\
  -DINSTALL_QHOUGH_NORMALS_PLUGIN=ON\
  -DINSTALL_QHPR_PLUGIN=ON\
  -DINSTALL_QM3C2_PLUGIN=ON\
  -DINSTALL_QPCV_PLUGIN=ON\
  -DINSTALL_QPHOTOSCAN_IO_PLUGIN=ON\
  -DINSTALL_QPOISSON_RECON_PLUGIN=ON\
  -DINSTALL_QSRA_PLUGIN=ON\
  -DINSTALL_QSSAO_PLUGIN=ON\
  -DBUILD_TESTING=ON\
  ${PWD}/..