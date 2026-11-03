"""Actual CUDA kernels and native IPC; skipped when the native helpers are absent."""
import numpy as np
import pytest
import torch
from PIL import Image

from dinov3_cases.gpu_geometry import LIBRARY, CudaPlaneFitter, prepare_rois
from dinov3_cases.inference.cat2real import prepare_cat2real_image

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or not LIBRARY.exists(),
                             reason='Thor CUDA helpers not built')


def test_cuda_consensus_rejects_outliers_and_refines_known_plane():
    rng=np.random.default_rng(2)
    points=rng.uniform(-.2,.2,(12000,3))
    points[:,2]=1+.2*points[:,0]+rng.normal(0,.0003,len(points))
    points[:1200,2]+=.03
    fitter=CudaPlaneFitter()
    try:
        plane,indices=fitter.fit(points,.005)
        assert len(indices)==10800
        expected=np.array([-.2,0,1.]);expected/=np.linalg.norm(expected)
        assert abs(plane[:3]@expected)>.99999
        assert np.max(abs(points[indices]@plane[:3]+plane[3]))<.002
        with pytest.raises(ValueError,match='finite Nx3'):
            fitter.fit(np.zeros((20,4)),.005)
        with pytest.raises(ValueError,match='degenerate_plane'):
            fitter.fit(np.ones((80,3)),.005)
    finally:fitter.close()


@pytest.mark.parametrize('shape',[(20,35),(35,20),(1,80),(80,1)])
def test_cuda_crops_preserve_center_padding_and_bilinear_pixels(shape):
    height,width=shape;rng=np.random.default_rng(3)
    pixels=rng.integers(0,256,(height,width,3),dtype=np.uint8)
    actual=prepare_rois(torch.from_numpy(pixels).cuda(),[[0,0,width,height]])[0].cpu().numpy()
    expected=np.asarray(prepare_cat2real_image(Image.fromarray(pixels)))
    assert abs(actual.astype(int)-expected.astype(int)).max()<=1


def test_cuda_roi_rejects_bounds_before_kernel_launch():
    pixels=torch.zeros((30,40,3),dtype=torch.uint8,device='cuda')
    with pytest.raises(ValueError,match='ROI bounds'):
        prepare_rois(pixels,[[0,0,50,30]])
    assert prepare_rois(pixels,[]).shape==(0,384,384,3)


def test_native_tracker_shared_rgb_agrees_with_legacy_pipe():
    from dinov3_cases.tracking.nvdcf import BRIDGE, NvDCF
    if not BRIDGE.exists():pytest.skip('Native tracker not built')
    pixels=np.zeros((480,640,3),np.uint8)
    pixels[100:300,150:350]=[200,60,20]
    results=[]
    for transport in ('pipe','shm'):
        tracker=NvDCF(transport=transport)
        try:
            initial=tracker.update(pixels,1,[dict(bbox=[150,100,350,300],label='box')])
            result=tracker.update(pixels,2)
            assert len(initial)==len(result)==1
            assert result[0]['label']=='box'
            results.append(result[0]['bbox'])
        finally:tracker.close()
        assert tracker.input_shm is None
    np.testing.assert_allclose(results[0],results[1],atol=2)
