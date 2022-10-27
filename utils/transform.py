
import numpy as np
import torch
from utils.rotation import get_random_rotations
from utils.cpp_funcs import furthest_point_sample_cpp


class ComposeAugment(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coord, feat, label):
        for t in self.transforms:
            coord, feat, label = t(coord, feat, label)
        return coord, feat, label


class RandomRotate(object):
    def __init__(self, mode='vertical', single_thread=True):
        self.mode = mode
        self.single_thread = single_thread

    def __call__(self, coord, feat, label):

        R = None
        if coord.shape[1] == 3:
            if self.mode == 'vertical':
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            elif self.mode == 'all':
                R = get_random_rotations(shape=None).astype(np.float32)  

        else:
            raise ValueError('Unsupported random rotation augment for point dimension: {:d}'.format(coord.shape[1]))

        if R is not None:
            if self.single_thread:
                coord = np.sum(np.expand_dims(coord, 2) * np.transpose(R), axis=1)
            else:
                coord = np.dot(coord, np.transpose(R))

        return coord, feat, label


class RandomScaleFlip(object):
    def __init__(self, scale=[0.9, 1.1], anisotropic=False, flip_p=[0.0, 0.0, 0.0]):
        self.scale = scale
        self.anisotropic = anisotropic
        self.flip_p = flip_p

    def __call__(self, coord, feat, label):
        
        if self.anisotropic:
            scale = np.random.uniform(self.scale[0], self.scale[1], coord.shape[1])
        else:
            scale = np.tile(np.random.uniform(self.scale[0], self.scale[1], 1), coord.shape[1])

        for ax, p in enumerate(self.flip_p):
            if np.random.rand() < p:
                scale[ax] *= -1.0

        coord *= scale.astype(np.float32)
        return coord, feat, label

        
class FloorCentering(object):
    """
    Centering the point cloud in the xy plane
    """
    def __init__(self, gravity_dim=2):
        self.gravity_dim = gravity_dim

    def __call__(self, coord, feat, label):
        coord -= np.mean(coord, axis=0, keepdims=True)
        coord[:, self.gravity_dim] -= np.min(coord[:, self.gravity_dim])
        return coord, feat, label

class UnitScaleCentering(object):
    """
    Centering the point cloud in the xy plane
    """
    def __init__(self, gravity_dim=2, height_feat_i=2):
        self.gravity_dim = gravity_dim
        self.height_feat_i = height_feat_i

    def __call__(self, coord, feat, label):

        # Get height before normalization
        height = coord[:, self.gravity_dim]
        feat[:, self.height_feat_i] = height - np.min(height)

        # Center
        coord -= np.mean(coord, axis=0, keepdims=True)

        # Normalize
        m = np.max(np.sqrt(np.sum(coord ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
        coord = coord / m

        return coord, feat, label

class RandomDrop(object):
    def __init__(self, p=0.15, fps=False):
        self.p = p
        self.fps = fps

    def __call__(self, coord, feat, label):

        if 0 < self.p < 1:
            N1 = coord.shape[0]
            N2 = int(np.ceil(N1 * (1 - self.p)))

            if self.fps:
                # Regular fps subsampling
                selection = np.random.permutation(N1)[:N2]
                selection = furthest_point_sample_cpp(torch.from_numpy(coord), new_n=N2)
            else:
                # Random drop of points
                selection = np.random.permutation(N1)[:N2]

            coord = coord[selection]
            if feat is not None:
                feat = feat[selection]
            if label is not None:
                label = label[selection]

        return coord, feat, label

class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip
        

    def __call__(self, coord, feat, label):
        if self.clip > 0: 
            jitter = np.random.randn(coord.shape[0], 3) * self.sigma
            coord += np.clip(jitter, -self.clip, self.clip)
        return coord, feat, label


class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None, uint8_colors=False):
        self.p = p
        self.blend_factor = blend_factor
        self.uint8_colors = uint8_colors

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            lo = np.min(feat[:, :3], 0, keepdims=True)
            hi = np.max(feat[:, :3], 0, keepdims=True)
            scale = 1.0 / (hi - lo)
            if self.uint8_colors:
                scale *= 255
            contrast_feat = (feat[:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            feat[:, :3] = (1 - blend_factor) * feat[:, :3] + blend_factor * contrast_feat
        return coord, feat, label


class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05, uint8_colors=False):
        self.p = p
        self.ratio = ratio
        self.uint8_colors = uint8_colors

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) * 2 - 1) * self.ratio
            if self.uint8_colors:
                tr *= 255
                feat[:, :3] = np.clip(tr + feat[:, :3], 0, 255)
            else:
                feat[:, :3] = np.clip(tr + feat[:, :3], 0, 1.0)
        return coord, feat, label


class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005, uint8_colors=False):
        self.p = p
        self.std = std
        self.uint8_colors = uint8_colors

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            noise = np.random.randn(feat.shape[0], 3) * self.std
            if self.uint8_colors:
                noise *= 255
                feat[:, :3] = np.clip(noise + feat[:, :3], 0, 255)
            else:
                feat[:, :3] = np.clip(noise + feat[:, :3], 0, 1.0)
        return coord, feat, label


class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2, uint8_colors=False):
        self.hue_max = hue_max
        self.saturation_max = saturation_max
        self.uint8_colors = uint8_colors

    def __call__(self, coord, feat, label):
        # Assume feat[:, :3] is rgb
        rgb = feat[:, :3]
        if not self.uint8_colors:
            rgb *= 255
        hsv = HueSaturationTranslation.rgb_to_hsv(rgb)
        hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feat[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        if not self.uint8_colors:
            feat[:, :3] *= 1 / 255
        return coord, feat, label


class RandomDropColor(object):

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            feat[:, :3] = 0
        return coord, feat, label


class RandomFullColor(object):
    """
    As in PointNext they do the random drop before Color normalize, it means the features 
    values after drop are not zeros but -2.XXXX. So here we use this function after normalize 
    and allow the colors to be randomly -2, 0 or 2 for each channel.
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            feat[:, :3] = 2 * np.random.randint(-1, 2, size=3)
        return coord, feat, label

    
class ChromaticNormalize(object):
    def __init__(self,
                 color_mean=[0.5136457, 0.49523646, 0.44921124],
                 color_std=[0.18308958, 0.18415008, 0.19252081]):
        self.color_mean = np.array(color_mean, dtype=np.float32)
        self.color_std = np.array(color_std, dtype=np.float32)

    def __call__(self, coord, feat, label):
        if np.mean(feat[:, :3]) > 1.0001:
            feat[:, :3] *= 1.0 / 255
        feat[:, :3] = (feat[:, :3] - self.color_mean) / self.color_std
        return coord, feat, label

class HeightNormalize(object):
    def __init__(self,
                 height_mean=[1.39467324],
                 height_std=[1.014554043]):
        self.height_mean = np.array(height_mean, dtype=np.float32)
        self.height_std = np.array(height_std, dtype=np.float32)

    def __call__(self, coord, feat, label):
        feat[:, 3] = (feat[:, 3] - self.height_mean) / self.height_std
        return coord, feat, label



