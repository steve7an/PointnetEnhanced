TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_INC='/usr/local/cuda-9.2/include'
CUDA_LIB='/usr/local/cuda-9.2/lib64/'

g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -I $TF_INC -I $CUDA_INC -lcudart -L $CUDA_LIB -L $TF_LIB -ltensorflow_framework -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 