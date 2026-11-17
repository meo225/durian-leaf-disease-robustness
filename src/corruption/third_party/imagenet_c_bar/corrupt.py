# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Trích từ facebookresearch/augmentation-corruption, imagenet_c_bar/corrupt.py, commit ba4d5a5e9132fe98fcd4be1ac90abedbbb188794.
# Chỉ giữ 10 họ có tag imagenet_c_bar. Thân hàm giữ nguyên, chỉ đổi cách import (xem NOTICE.md).

import numpy as np
from scipy.fftpack import ifft2
from scipy.ndimage import gaussian_filter

from .base import Transform
from .image_utils import bilinear_interpolation
from .perlin_noise import PerlinNoiseGenerator


def int_parameter(level, maxval):
  return int(level * maxval / 10)

def float_parameter(level, maxval):
  return float(level) * maxval / 10.

class SingleFrequencyGreyscale(Transform):

    name = 'single_frequency_greyscale'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        freq_mag = np.random.uniform(low=-np.pi, high=np.pi)
        freq_2 = np.random.uniform(low=-abs(freq_mag), high=abs(freq_mag))
        freq = np.array([freq_mag, freq_2])[np.random.permutation(2)]
        phase = np.random.uniform(low=0, high=2*np.pi)
        intensity = float_parameter(self.severity, 196)
        return { 'freq' : freq, 'phase' : phase, 'intensity' : intensity}

    def transform(self, image, freq, phase, intensity):
        noise = np.array([[np.sin(x * freq[0] + y * freq[1] + phase)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = np.stack((noise, noise, noise), axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

    def convert_to_numpy(self, params):
        return np.array(params['freq'].tolist() + [params['phase'], params['intensity']])

    def convert_from_numpy(self, numpy_record):
        return {'freq' : numpy_record[0:2],
                'phase' : numpy_record[2],
                'intensity' : numpy_record[3]
                }

class CocentricSineWaves(Transform):

    name = 'cocentric_sine_waves'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        offset = np.random.uniform(low=0, high=self.im_size, size=2)
        freq = np.random.uniform(low=0, high=10)
        amplitude = np.random.uniform(low=0, high=self.im_size/10)
        ring_width = np.random.uniform(low=0, high=self.im_size/10)
        intensity = [float_parameter(self.severity, 128) for i in range(3)]

        return { 'offset' : offset,
                 'freq' : freq,
                 'amplitude' : amplitude,
                 'ring_width' : ring_width,
                 'intensity' : intensity
                }

    def transform(self, image, offset, freq, amplitude, ring_width, intensity):

        def calc_intensity(x, y, x0, y0, freq, amplitude, ring_width):
            angle = np.arctan2(x-x0, y-y0) * freq
            distance = ((np.sqrt((x-x0)**2 + (y-y0)**2) + np.sin(angle) * amplitude) % ring_width) / ring_width
            distance -= 1/2
            return distance

        noise = np.array([[calc_intensity(x, y, offset[0], offset[1], freq, amplitude, ring_width)\
                    for x in range(self.im_size)] for y in range(self.im_size)])
        noise = np.stack((intensity[0] * noise, intensity[1] * noise, intensity[2] * noise), axis=2)

        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def convert_to_numpy(self, params):
        return np.array(params['offset'].tolist() + [params['freq'], params['amplitude'], params['ring_width']] + params['intensity'])

    def convert_from_numpy(self, numpy_record):
        return {'offset' : numpy_record[0:2].tolist(),
                'freq' : numpy_record[2],
                'amplitude' : numpy_record[3],
                'ring_width' : numpy_record[4],
                'intensity' : numpy_record[4:7].tolist()
                }
        

class PlasmaNoise(Transform):

    name = 'plasma_noise'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        time = np.random.uniform(low=0.0, high=6*np.pi)
        iterations = np.random.randint(low=4, high=7)
        sharpness = np.random.uniform(low=0.5, high=1.0)
        scale = np.random.uniform(low=0.075, high=0.2) * self.im_size
        intensity = float_parameter(self.severity,64)
        return {'time' : time, 'iterations' : iterations, 'sharpness' : sharpness,
                'scale' : scale, 'intensity' : intensity}

    def transform(self, image, time, iterations, sharpness, scale, intensity):

        def kernel(x, y, rand, iters, sharp, scale):
            x /= scale
            y /= scale
            i = np.array([1.0, 1.0, 1.0, 0.0])
            for s in range(iters):
                r = np.array([np.cos(y * i[0] - i[3] + rand / i[1]), np.sin(x * i[0] - i[3] + rand / i[1])]) / i[2]
                r += np.array([-r[1],r[0]]) * 0.3
                x += r[0]
                y += r[1]
                i *= np.array([1.93, 1.15, (2.25 - sharp), rand * i[1]])
            r = np.sin(x - rand)
            b = np.sin(y + rand)
            g = np.sin((x + y + np.sin(rand))*0.5)
            return [r,g,b]


        noise = np.array([[kernel(x,y, time, iterations, sharpness, scale)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip((1-intensity/255) * image + intensity * noise, 0, 255).astype(np.uint8)

class Sparkles(Transform):

    name = 'sparkles'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        centers = np.random.uniform(low=0, high=self.im_size, size=(5, 2))
        radii = np.array([float_parameter(self.severity, 0.1)\
                for i in range(5)]) * self.im_size
        amounts = np.array([50 for i in range(5)])
        color = np.array([255, 255, 255])
        randomness = 25
        seed = np.random.randint(low=0, high=2**32)
        nrays = np.random.randint(low=50, high=200, size=5)

        return {'centers' : centers, 'radii' : radii, 'color' : color, 'randomness' : randomness,
                'seed' : seed, 'nrays' : nrays, 'amounts' : amounts
                }

    def transform(self, image, centers, radii, nrays, amounts, color, randomness, seed):

        def kernel(point, value, center, radius, ray_lengths, amount, color):
            rays = len(ray_lengths)
            dp = point - center
            dist = np.linalg.norm(dp)
            angle = np.arctan2(dp[1], dp[0])
            d = (angle + np.pi) / (2 * np.pi) * rays
            i = int(d)
            f = d - i 

            if radius != 0:
                length = ray_lengths[i % rays] + f * (ray_lengths[(i+1) % rays] - ray_lengths[i % rays])
                g = length**2 / (dist**2 + 1e-4)
                g = g ** ((100 - amount) / 50.0)
                f -= 0.5
                f = 1 - f**2
                f *= g
            f = np.clip(f, 0, 1)
            return value + f * (color - value)

        random_state = np.random.RandomState(seed=seed)
        for center, rays, amount, radius in zip(centers, nrays, amounts, radii):
            ray_lengths = [max(1,radius + randomness / 100.0 * radius * random_state.randn())\
                for i in range(rays)]

            image = np.array([[kernel(np.array([y,x]), image[y,x,:].astype(np.float32), center, radius, ray_lengths, amount, color)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(image, 0, 255).astype(np.uint8)

class InverseSparkles(Transform):

    name = 'inverse_sparkles'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        center = np.random.uniform(low=0.25, high=0.75, size=2) * self.im_size
        radius = 0.25 * self.im_size
        amount = 100
        amount = float_parameter(self.severity, 65)
        amount = 100 - amount
        color = np.array([255, 255, 255])
        randomness = 25
        seed = np.random.randint(low=0, high=2**32)
        rays = np.random.randint(low=50, high=200)

        return {'center' : center, 'radius' : radius, 'color' : color, 'randomness' : randomness,
                'seed' : seed, 'rays' : rays, 'amount' : amount
                }

    def transform(self, image, center, radius, rays, amount, color, randomness, seed):

        def kernel(point, value, center, radius, ray_lengths, amount, color):
            rays = len(ray_lengths)
            dp = point - center
            dist = np.linalg.norm(dp)
            angle = np.arctan2(dp[1], dp[0])
            d = (angle + np.pi) / (2 * np.pi) * rays
            i = int(d)
            f = d - i 

            if radius != 0:
                length = ray_lengths[i % rays] + f * (ray_lengths[(i+1) % rays] - ray_lengths[i % rays])
                g = length**2 / (dist**2 + 1e-4)
                g = g ** ((100 - amount) / 50.0)
                f -= 0.5
                f = 1 - f**2
                f *= g
            f = np.clip(f, 0, 1)
            return color + f * (value - color)

        random_state = np.random.RandomState(seed=seed)
        ray_lengths = [radius + randomness / 100.0 * radius * random_state.randn()\
                for i in range(rays)]

        out = np.array([[kernel(np.array([y,x]), image[y,x,:].astype(np.float32), center, radius, ray_lengths, amount, color)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(out, 0, 255).astype(np.uint8)

class PerlinNoise(Transform):

    name = 'perlin_noise'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        m = np.array([[1,0],[0,1]]) / (32 * self.im_size / 224)
        turbulence = 16.0
        gain = 0.5
        bias = 0.5
        alpha = float_parameter(self.severity, 0.50)
        seed = np.random.randint(low=0, high=2**32)
        return {'m': m, 'turbulence' : turbulence, 'seed': seed,
                'gain': gain, 'bias': bias, 'alpha': alpha}

    def transform(self, image, m, turbulence, seed, gain, bias, alpha):
        
        random_state = np.random.RandomState(seed=seed)
        noise = PerlinNoiseGenerator(random_state)

        def kernel(point, m, turbulence, gain, bias):
            npoint = np.matmul(point, m)
            f = noise.turbulence(npoint[0], npoint[1], turbulence)\
                    if turbulence != 1.0 else noise.noise(npoint[0], npoint[1])
            f = gain * f + bias
            return np.clip(np.array([f,f,f]),0,1.0)

        noise = np.array([[kernel(np.array([y,x]),m,turbulence,gain, bias) for x in range(self.im_size)]\
                for y in range(self.im_size)])
        out = (1 - alpha) * image.astype(np.float32) + 255 * alpha * noise
        return np.clip(out, 0, 255).astype(np.uint8)

class BrownishNoise(Transform):

    name = 'brownish_noise'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        intensity = float_parameter(self.severity, 64)

        return {'seed' : seed, 'intensity' : intensity}

    def gen_noise(self, random_state):
        center = self.im_size / 2
        power = np.array([[1/(np.linalg.norm(np.array([x,y])-center)**2+1)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        phases = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size, self.im_size//2))
        if self.im_size % 2 == 0:
            phases = np.concatenate((phases, phases[::-1,::-1]), axis=1)
        else:
            center_freq = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size//2, 1))
            center_freq = np.concatenate((center_freq, np.array([[0.0]]), center_freq[::-1,:]), axis=0)
            phases = np.concatenate((phases, center_freq, phases[::-1,::-1]), axis=1)
        fourier_space_noise = power * (np.cos(phases) + np.sin(phases) * 1j)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=0)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=1)


        noise = np.real(ifft2(fourier_space_noise))
        noise = noise / np.std(noise)
        return noise

    def transform(self, image, seed, intensity):
        random_state = np.random.RandomState(seed=seed)
        noise = np.stack([self.gen_noise(random_state) for i in range(3)],axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

class CheckerBoardCutOut(Transform):

    name = 'checkerboard_cutout'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        angle = np.random.uniform(low=0, high=2*np.pi)
        scales = np.maximum(np.random.uniform(low=0.1, high=0.25) * self.im_size, 1)
        scales = (scales, scales)
        fraction = float_parameter(self.severity, 1.0)
        seed = np.random.randint(low=0, high=2**32)

        return {'angle' : angle, 'scales' : scales, 'fraction' : fraction, 'seed' : seed}

    def transform(self, image, scales, angle, fraction, seed):
        random_state = np.random.RandomState(seed=seed)
        grid = random_state.uniform(size=(int(4*self.im_size//scales[0]), int(4*self.im_size//scales[1]))) < fraction
        
        def mask_kernel(point, scales, angle, grid):
            nx = (np.cos(angle) * point[0] + np.sin(angle) * point[1]) / scales[0]
            ny = (-np.sin(angle) * point[0] + np.cos(angle) * point[1]) / scales[1]
            return (int(nx % 2) != int(ny % 2)) or not grid[int(nx),int(ny)]

        out = np.array([[image[y,x,:] if mask_kernel([y,x], scales, angle, grid) else np.array([128,128,128])\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip(out, 0, 255).astype(np.uint8)

class BlueNoiseSample(Transform):

    name = 'blue_noise_sample'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        threshold = float_parameter(self.severity, 3.0) - 2.5

        return {'seed' : seed, 'threshold' : threshold}

    def transform(self, image, seed, threshold):
        random_state = np.random.RandomState(seed=seed)

        center = self.im_size / 2
        power = np.array([[np.linalg.norm(np.array([x,y])-center)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        phases = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size, self.im_size//2))
        if self.im_size % 2 == 0:
            phases = np.concatenate((phases, phases[::-1,::-1]), axis=1)
        else:
            center_freq = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size//2, 1))
            center_freq = np.concatenate((center_freq, np.array([[0.0]]), center_freq[::-1,:]), axis=0)
            phases = np.concatenate((phases, center_freq, phases[::-1,::-1]), axis=1)
        fourier_space_noise = power * (np.cos(phases) + np.sin(phases) * 1j)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=0)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=1)


        noise = np.real(ifft2(fourier_space_noise))
        noise = noise / np.std(noise)
        mask = noise > threshold
        out = image * mask.reshape(self.im_size, self.im_size, 1)


        return np.clip(out, 0, 255).astype(np.uint8)

class CausticRefraction(Transform):

    name = 'caustic_refraction'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        time = np.random.uniform(low=0.5, high=2.0)
        size = np.random.uniform(low=0.75, high=1.25) * self.im_size
        #size = self.im_size
        eta = 4.0
        lens_scale = float_parameter(self.severity, 0.5*self.im_size)
        lighting_amount = float_parameter(self.severity, 2.0)
        softening = 1

        return { 'time' : time, 'size' : size, 'eta' : eta, 'lens_scale' : lens_scale, 'lighting_amount': lighting_amount, 'softening' : softening}

    def transform(self, image, time, size, eta, lens_scale, lighting_amount, softening):

        def caustic_noise_kernel(point, time, size):
            point = point / size
            p = (point % 1) * 6.28318530718 - 250

            i = p.copy()
            c = 1.0
            inten = 0.005

            for n in range(5):
                t = time * (1.0 - (3.5 / (n+1)))
                i = p + np.array([np.cos(t-i[0])+np.sin(t+i[1]),np.sin(t-i[1])+np.cos(t+i[0])])
                length = np.sqrt((p[0] / (np.sin(i[0]+t)/inten))**2 + (p[1] / (np.cos(i[1]+t)/inten))**2)
                c += 1.0/length

            c /= 5.0
            c = 1.17 - c ** 1.4
            color = np.clip(np.abs(c) ** 8.0, 0, 1) 
            return np.array([color, color, color])


        def refract(incident, normal, eta):
            if np.abs(np.dot(incident, normal)) >= 1.0 - 1e-3:
                return incident
            angle = np.arccos(np.dot(incident, normal))
            out_angle = np.arcsin(np.sin(angle) / eta)
            out_unrotated = np.array([np.cos(out_angle), np.sin(out_angle), 0.0])
            spectator_dim = np.cross(incident, normal)
            spectator_dim /= np.linalg.norm(spectator_dim)
            orthogonal_dim = np.cross(normal, spectator_dim)
            rotation_matrix = np.stack((normal, orthogonal_dim, spectator_dim), axis=0)
            return np.matmul(np.linalg.inv(rotation_matrix), out_unrotated)

        def luma_at_offset(image, origin, offset):
            pixel_value = image[origin[0]+offset[0], origin[1]+offset[1], :]\
                    if origin[0]+offset[0] >= 0 and origin[0]+offset[0] < image.shape[0]\
                    and origin[1]+offset[1] >= 0 and origin[1]+offset[1] < image.shape[1]\
                    else np.array([0.0,0.0,0])
            return np.dot(pixel_value, np.array([0.2126, 0.7152, 0.0722]))

        def luma_based_refract(point, image, caustics, eta, lens_scale, lighting_amount):
            north_luma = luma_at_offset(caustics, point, np.array([0,-1]))
            south_luma = luma_at_offset(caustics, point, np.array([0, 1]))
            west_luma = luma_at_offset(caustics, point, np.array([-1, 0]))
            east_luma = luma_at_offset(caustics, point, np.array([1,0]))

            lens_normal = np.array([east_luma - west_luma, south_luma - north_luma, 1.0])
            lens_normal = lens_normal / np.linalg.norm(lens_normal)

            refract_vector = refract(np.array([0.0, 0.0, 1.0]), lens_normal, eta) * lens_scale
            refract_vector = np.round(refract_vector, 3)

            out_pixel = bilinear_interpolation(image, point+refract_vector[0:2])
            out_pixel += (north_luma - south_luma) * lighting_amount
            out_pixel += (east_luma - west_luma) * lighting_amount

            return np.clip(out_pixel, 0, 1)

        noise = np.array([[caustic_noise_kernel(np.array([y,x]), time, size)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = gaussian_filter(noise, sigma=softening)

        image = image.astype(np.float32) / 255
        out = np.array([[luma_based_refract(np.array([y,x]), image, noise, eta, lens_scale, lighting_amount)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip((out * 255).astype(np.uint8), 0, 255)


IMAGENET_C_BAR_FAMILIES = {
    cls.name: cls
    for cls in (
        SingleFrequencyGreyscale, CocentricSineWaves, PlasmaNoise, Sparkles, InverseSparkles,
        PerlinNoise, BrownishNoise, CheckerBoardCutOut, BlueNoiseSample, CausticRefraction,
    )
}
