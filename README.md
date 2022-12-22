# Training Deep Nets with Sublinear Memory Cost
![Figure 1](image/overview.png)
> Training very deep neural networks requires a lot of memory. Using the tools in this package, developed jointly by Tim Salimans and Yaroslav Bulatov, you can trade off some of this memory usage with computation to make your model fit into memory more easily. For feed-forward models we were able to fit more than 10x larger models onto our GPU, at only a 20% increase in computation time.

This code is the Pytorch implementation of the paper [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/pdf/1604.06174.pdf).

## Requirements
* Pytorch
* torchvision

## Run
You can run MNIST experiments by running following command. 
```
python mnist_full.py
```

## Citation

If you use this code in your work, please cite this paper:  

```none
@article{chen2016training,
  title={Training deep nets with sublinear memory cost},
  author={Chen, Tianqi and Xu, Bing and Zhang, Chiyuan and Guestrin, Carlos},
  journal={arXiv preprint arXiv:1604.06174},
  year={2016}
}
```
