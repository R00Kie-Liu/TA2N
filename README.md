# TA2N: Two-Stage Action Alignment Network for Few-Shot Action Recognition

The core code of TA2N has been released. More details will be published soon.

## Usage in FSL model:
First, obtain video feature embeddings through the Resnet-50 and transpose into (N, T, C, H, W) shape.

```python
support = torch.rand(5, 2048, 8, 7, 7).cuda()
query = torch.rand(5, 2048, 8, 7, 7).cuda()

```
Then, feed the support and query feature into ta2n
```python
ta2n = TA2N(T=8,shot=1, dim=(2048,2048),first_stage=TTM, second_stage=ACM).cuda()
pairs, offsets = ta2n(support, query)
```

You can split aligned query and support features from paired feature:

```
aligned_support, aligned_query = pairs[:,:,:2048,...],pairs[:,:,2048:,...]
```