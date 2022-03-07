# matchNetPytorch
reproduction of matchNet using pytorch, since the original implementation is based on caffe
and some newer reproductions are based on tensorflow-keras, I reproduce matchNet using pytorch
to ease my work

# defference compared to original implementation

I provide different feature net usiing mobileNetV3' block, in experiments, feature net constructed 
by mobileNetV3 block has better training efficiency and performace.

Notice that, I use accuracy( correct predictions / total predictions ) instead of FPR@95recall

# Acknowledgments
Thanks the author who provide the mobileNetV3 model implementation at https://github.com/d-li14/mobilenetv3.pytorch
