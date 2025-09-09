# Object Tracking: A Quick Overview

## Multi Object Tracking

### End-to-End Tracking

**MOTR Family**

- [MeMOTR](https://arxiv.org/abs/2307.15700)
- [MOTRv3](https://arxiv.org/abs/2305.14298) and [CO-MOT](https://arxiv.org/abs/2305.12724): Improvement on [MOTR](https://arxiv.org/abs/2105.03247) by increasing detection objects in the loss term as extra supervision.
- [MO-YOLO](https://arxiv.org/abs/2310.17170): : Efficient (fast training on 1 2080 ti GPU, 8 hours) YOLO with transformer in the encoder (RT-YOLO) and MOTR in the decoder.

### Point Tracking (Track any point)

- CoTracker3  (Andrea Vedaldi)
- MVTracker (Marc Pollefeys): for multi-view 3D point tracking. Similar as CoTracker but it works in 3D. 

### Others

- NOOUGAT: online/offline graph-based learned associations