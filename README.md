# theano_newbackend_debug
This repo is dedicated to debug of theano on newbackend

The issue is related to Theano's optimization and new backend at GPU. The demo code will run on old backend (CPU and GPU), and new backend at CPU, but won't on new backend at GPU, error raised as
```
ValueError: GpuElemwise. Input dimension mis-match. Input 1 (indices start at 0) has shape[0] == 4, but the output's size on that axis is 31.
```
Seemed there was tensor reshaped wrongly inside the GPU.

To run the demo code, you'll need [Lasagne_CTC](https://github.com/david-leon/Lasagne_CTC) and [Lasagne_Ext](https://github.com/david-leon/Lasagne_Ext)

