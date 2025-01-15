### 🧰 Common Time Series Data Augmentation Techniques
Beyond guided warping, several standard augmentation methods are widely used in time series analysis

- **Jittering**: Adding random noise to the time series
- **Scaling**: Multiplying the time series by a random scalar to change its amplitude
- **Permutation**: Randomly shuffling segments of the time series
- **Time Warping**: Stretching or compressing the time intervals in the series
- **Magnitude Warping**: Applying smooth, nonlinear transformations to the amplitude
- **Window Slicing**: Extracting random sub-sequences from the time series
- **Window Warping**: Randomly stretching or compressing specific windows within the time series
- **Flipping**: Reversing the time series sequence
- **Mixup**: Combining two time series by weighted averaging
- **Frequency Domain Augmentation**:Altering the frequency components of the time series

---

### 🧪 Python Implementations
Below are Python implementations for some of these augmentation technique:

#### Jittering

```python
import numpy as np

def jitter(x, sigma=0.03):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
```


#### Scaling

```python
def scaling(x, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1],))
    return x * factor
```


#### Permutation

```python
def permutation(x, max_segments=5):
    orig_steps = np.arange(x.shape[0])
    num_segs = np.random.randint(1, max_segments)
    split_points = np.array_split(orig_steps, num_segs)
    np.random.shuffle(split_points)
    permuted = np.concatenate(split_points)
    return x[permuted]
```


#### Time Warping

```python
from scipy.interpolate import CubicSpline

def time_warp(x, sigma=0.2):
    orig_steps = np.arange(x.shape[0])
    random_warp = np.random.normal(loc=1.0, scale=sigma, size=x.shape[0])
    warp_steps = np.cumsum(random_warp)
    warp_steps = (warp_steps - warp_steps.min()) / (warp_steps.max() - warp_steps.min()) * (x.shape[0] - 1)
    cs = CubicSpline(warp_steps, x, axis=0)
    return cs(orig_steps)
```


#### Magnitude Warping

```python
def magnitude_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2,))
    warp_steps = np.linspace(0, x.shape[0]-1, num=knot+2)
    cs = CubicSpline(warp_steps, random_warps)
    return x * cs(orig_steps).reshape(-1, 1)
```


#### Window Slicing

```python
def window_slice(x, reduce_ratio=0.9):
    target_len = int(np.ceil(reduce_ratio * x.shape[0]))
    start = np.random.randint(0, x.shape[0] - target_len)
    return x[start:start+target_len]
```


#### Window Warping

```python
def window_warp(x, window_ratio=0.1, scales=[0.5, 2.0]):
    warp_size = int(np.ceil(window_ratio * x.shape[0]))
    start = np.random.randint(0, x.shape[0] - warp_size)
    window = x[start:start+warp_size]
    scale = np.random.choice(scales)
    window = resample(window, int(warp_size * scale))
    warped = np.concatenate((x[:start], window, x[start+warp_size:]))
    return resample(warped, x.shape[0])
```


*Note: The `resample` function can be imported from `scipy.signal`.*

---

### 🔗 Additional Resources

- Time Series Data Augmentation for Neural Networks by Time Warping
with a Discriminative Teacher: [paper](https://arxiv.org/pdf/2004.08780), An empirical survey of data augmentation for time series classification with neural networks: [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8282049), [code](https://github.com/uchidalab/time_series_augmentation)

- **Tsaug Library**: An open-source Python package for time series augmentation, offering a variety of augmentation techniques: [code](https://github.com/arundo/tsaug)

