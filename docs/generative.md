<!-- ### Diffusion-Inspired Techniques for Enhanced 3D Reconstruction

[FlowR, Marc Pollefeys](https://arxiv.org/pdf/2504.01647), 
FlowR addresses the challenge of sparse 3D reconstructions by synthesizing high-quality, dense novel views from a limited set of input perspectives. The paper introduces a flow-matching model designed to predict the transformation required to generate renderings consistent with a dense 3D reconstruction, effectively refining input views before they are integrated into broader 3D pipelines.

[Difix3D, Sanja Fidler](https://arxiv.org/pdf/2503.01774)
Difix3D presents a novel framework for multi-view 3D reconstruction by leveraging a diffusion-inspired approach that maintains strong consistency with input images while enabling the generation of diverse, plausible outputs. This method is particularly well-suited for scenarios with inherently ambiguous inputs where traditional single-solution reconstructions may fall short.


## usupervised data (self-training)

SMURFL:  unsupervised optical flow
DepthG, STEGO: unsupervised semantic segementation
CutLER: unsupervised instant segmentation
U2Seg (CutLER + STEGO), CUPS (Cremers, SMURF + DepthG): unsupervised panoptic segmentation

AnyCam, Cremers
Panoptic Lidar: Taixe -->

## Generative Models

#### Diffusion model related

- Diffusion model: [DDPM, Ho 2020](https://arxiv.org/abs/2006.11239);[Sohl-Dickstein 2015](https://arxiv.org/abs/1503.03585); Score-based, [Song 2019](https://arxiv.org/abs/1907.05600), [Song 2021](https://arxiv.org/abs/2011.13456); Tuturial [AssemblyAI](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction); [Li'Ling](https://lilianweng.github.io/posts/2021-07-11-diffusion-models)
- Flow matching /  [Lipman](https://arxiv.org/abs/2210.02747), [Lipman Guide and Code](https://arxiv.org/abs/2412.06264), [Introduction by Cambridge](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html); [GMFlow](https://arxiv.org/abs/2504.05304)
- Rectified Flow: [Liu](https://arxiv.org/abs/2209.03003); [Liu](https://arxiv.org/pdf/2209.14577)
- Normalizing Flow

#### TO be organized

### Code: 

- [Pyramid-Flow](https://github.com/jy0205/Pyramid-Flow/blob/main/pyramid_dit/mmdit_modules/modeling_pyramid_mmdit.py)
- [Flow Matching](https://github.com/facebookresearch/flow_matching)
- [Stable Diffusion 3.5](https://github.com/Stability-AI/sd3.5)

### New works

- Kaiming He: [Mean Flows](https://arxiv.org/pdf/2505.13447) (similar to [Align Your Flow](https://arxiv.org/pdf/2506.14603)), [Dispersive Loss](https://arxiv.org/pdf/2506.09027)
