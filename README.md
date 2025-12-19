## How to build and run code

**You need to set up the environment before running the code, see the instructions below**  
After you are done setting up the environment, running the code is easy.  
To run the cuda version, just call the shell script below. Ensure you are in the /deform-attn-cuda/ folder first.
```bash
./run_cu.sh
```
To call the python version, stay in the /deform-attn-cuda/ folder and call the profiler script:  
```python
python dat_torch/prof_dat.py
```


## How to run correctness tests

## How to run performance/speedup tests

## How to run Layernorm/GELU unit tests (both speedup and correctness)
The unit tests for Layernorm and GELU functions are in a seperate folder. To run them, go to `layernorm-GELU-tests` directory and run the following bash command, replacing test1 with whatever test case, from 1-10, you want to use. 
```bash
bash ln_run_cu.sh test1
```
Each of these testcases will repeatedly run some variations of the layernorm and GELU functions (unfused with no shuffling, unfused with shuffle, fused and optimized) on a single testcase to measure performance, and will also compare the outputs with a ground truth value computed from the PyTorch implementation of layernorm and GELU to test for accuracy.

## Environment setup
All of this code was built and tested while SSH'd into a CSE Labs machine, so make sure you are SSH'd into a cuda machine. We used `csel-cuda-03.cselabs.umn.edu`.

The python portion of this project requires setting up a specific conda environment.  
Further, the cuda portion of this project requires cuDNN and cnpy. Neither of these are installed by default on the lab machines so you will need to install them yourself. 

#### Setting up Conda
Do the following steps in the /deform-attn-cuda/ folder, copy-pasting each block into the terminal and running it, in order:
```python
conda create -n dat python=3.9
```
```python
conda activate dat
```
```python
pip install \
torch==1.11.0 \
torchvision==0.12.0 \
numpy==1.20.3 \
timm==0.5.4 \
einops==0.6.1 \
PyYAML \
yacs \
termcolor
```
```python
pip3 install natten==0.14.6+torch1110cpu -f https://whl.natten.org/old
```

#### Setting up cuDNN
Go to the cuDNN archive: https://developer.nvidia.com/rdp/cudnn-archive.  
Then find and download the appropriate cuDNN version for the lab machines and extract it:  
```Cudnn-linux-x86_64-8.8.0.121_cuda12-archive.tar```  


Now you need to set up the environment variables that the build script will use to find cuDNN. Run the following lines in the terminal one at a time. Note that the changes will disapear when you close the terminal - if you want them to persist then copy them at the end of the ./bashrc script to run them every time you open the kernel.
```bash
export CUDNN_HOME=$HOME/local/cudnn/cudnn-linux-x86_64-8.8.0.121_cuda12-archive
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}
export CPATH=$CUDNN_HOME/include:${CPATH:-}
```


#### Setting up cnpy
Cnpy is a utility that allows reading numpy files with C++, available at the github repo https://github.com/rogersce/cnpy. Just clone it into the /deform-attn-cuda/ folder.
```git
git clone https://github.com/rogersce/cnpy.git
```


### Reference material
Repo for the base DAT code in Python: https://github.com/LeapLabTHU/DAT  
Paper by the same team that made the repo, describing the code in more depth: https://arxiv.org/pdf/2201.00520  