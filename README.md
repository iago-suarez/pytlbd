# pytlbd
Python transparent bindings for Line Band Descriptor

The repository contains a C++ implementation with python bindings of the method:

```
Zhang, L., & Koch, R. (2013). An efficient and robust line segment matching approach based on LBD descriptor 
and pairwise geometric consistency. Journal of Visual Communication and Image Representation, 24(7), 794-805.
```

The code is based on the original one provided by Lilian Zhang. The re-implementation substitute the computer 
vision library BIAS by OpenCV and uses modern C++14 structures instead of C ones. Optionally, ARPACK can be used to 
optimize the line matching part.  

### Download
```
git clone git@github.com:iago-suarez/pytlbd.git
cd pytldb
git submodule update --init --recursive 
```

### Build

You need to have installed OpenCV in your system or rather modify the setup.py file to
include its installation directory.

```
git clone --recursive https://github.com/iago-suarez/pytlbd.git
cd pytlbd
pip3 install -r requirements.txt
pip3 install .
```

To compile the C++ code:

```
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_ARPACK=ON ..
make -j
```
