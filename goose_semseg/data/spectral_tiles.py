"""On-demand source-grid tiles with train-only sensor normalization.

Input slots: seven normalized source pages, two raw-page ratio proxies.
P1 occupies B/G/R slots only; unavailable spectral/ratio slots stay zero.
Validation/test tiles never overlap and padded pixels are ignored.
"""
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class SpectralTileDataset(Dataset):
    def __init__(self, root, split, tile_size=768, flip_prob=0.5, rare_prob=0.7,
                 class_sampling_mode='global'):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.source = Path(self.manifest['source'])
        self.records = [r for r in self.manifest['records'] if r['split']==split]
        self.split, self.tile_size = split, int(tile_size)
        self.flip_prob, self.rare_prob = flip_prob, rare_prob
        if class_sampling_mode not in ('global', 'within_image'):
            raise ValueError(f'Unknown class sampling mode: {class_sampling_mode}')
        if not 0 <= rare_prob <= 1:
            raise ValueError('rare_prob must lie in [0, 1]')
        self.class_sampling_mode = class_sampling_mode
        self.normalization = self.manifest['normalization']
        # Overlapping annotations are rasterized exactly as in the previous converter.
        class_names = self.manifest.get('class_names')
        if self.manifest.get('annotation_label_key') == 'label_id' and not class_names:
            raise ValueError('Native label_id manifests must declare class_names')
        # Backward compatibility for legacy source manifests and small synthetic tests.
        self.num_classes = len(class_names) if class_names else 11
        self.class_members = {
            i: [j for j,r in enumerate(self.records) if i in r['classes_present']]
            for i in range(self.num_classes)
        }
        self.active_classes = [i for i, indices in self.class_members.items() if indices]
        self.tiles = []
        if split != 'train':
            for j,r in enumerate(self.records):
                for top in range(0, r['height'], self.tile_size):
                    for left in range(0, r['width'], self.tile_size):
                        self.tiles.append((j, left, top))

    def __len__(self):
        return len(self.records) if self.split == 'train' else len(self.tiles)

    @lru_cache(maxsize=1)
    def _source_pixels(self, index):
        r = self.records[index]
        with Image.open(self.source/r['source_image']) as im:
            if r['sensor']=='P1':
                return np.array(im.convert('RGB'))
            if getattr(im,'n_frames',1)!=7:
                raise ValueError('Unexpected Altum page count')
            bands=[]
            for page in range(7):
                im.seek(page)
                bands.append(np.array(im))
            return np.stack(bands)

    @lru_cache(maxsize=2)
    def _mask(self, index):
        r = self.records[index]
        mask = Image.new('L', (r['width'], r['height']), 255)
        draw = ImageDraw.Draw(mask)
        for ann in sorted(r['annotations'], key=lambda a: a.get('area',0), reverse=True):
            if self.manifest.get('annotation_label_key') == 'label_id':
                cid = int(ann['label_id'])
                if cid != 255 and not 0 <= cid < self.num_classes:
                    raise ValueError(f'Invalid native label_id={cid} for {self.num_classes} classes')
            else:
                cid = ann['category_id'] - 1 - int(ann['category_id'] > 11)
            for polygon in ann.get('segmentation', []):
                if len(polygon) >= 6:
                    draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=cid)
        return np.array(mask)

    def __getitem__(self, index):
        target = None
        if self.split == 'train':
            if random.random() < self.rare_prob:
                if self.class_sampling_mode == 'global':
                    # Legacy protocol: choose a class, then replace the image.
                    target = random.choice(self.active_classes)
                    index = random.choice(self.class_members[target])
                else:
                    # Preserve shuffled image exposure; only bias the crop location.
                    present = np.unique(self._mask(index))
                    present = present[present != 255].tolist()
                    if present:
                        target = random.choice(present)
            r = self.records[index]
            left = random.randint(0, max(0,r['width']-self.tile_size))
            top = random.randint(0, max(0,r['height']-self.tile_size))
            if target is not None:
                yy,xx = np.where(self._mask(index)==target)
                if len(xx):
                    k = random.randrange(len(xx))
                    left = min(max(0, int(xx[k])-random.randrange(self.tile_size)), max(0,r['width']-self.tile_size))
                    top = min(max(0, int(yy[k])-random.randrange(self.tile_size)), max(0,r['height']-self.tile_size))
        else:
            index,left,top = self.tiles[index]
            r = self.records[index]
        width = min(self.tile_size,r['width']-left)
        height = min(self.tile_size,r['height']-top)
        label = np.full((self.tile_size,self.tile_size),255,dtype=np.uint8)
        label[:height,:width] = self._mask(index)[top:top+height,left:left+width]
        bands = np.zeros((7,height,width),dtype=np.float32)
        pixels = self._source_pixels(index)
        sh,sw = pixels.shape[-2:] if r['sensor']=='Altum' else pixels.shape[:2]
        sx,sy = sw/r['width'],sh/r['height']
        x0,y0,x1,y1 = (round(v) for v in (left*sx,top*sy,(left+width)*sx,(top+height)*sy))
        if r['sensor']=='Altum':
            for i in range(7):
                page = pixels[i,y0:y1,x0:x1].astype(np.float32)
                if page.shape != (height,width):
                    page = np.asarray(Image.fromarray(page).resize((width,height),Image.Resampling.BILINEAR))
                bands[i] = page
        else:
            rgb = Image.fromarray(pixels[y0:y1,x0:x1]).resize((width,height),Image.Resampling.BILINEAR)
            bands[:3] = np.asarray(rgb,dtype=np.float32)[...,::-1].transpose(2,0,1)*257
        stats = self.normalization[r['sensor']]
        low = np.asarray(stats['low'],dtype=np.float32)[:,None,None]
        scale = np.asarray(stats['scale'],dtype=np.float32)[:,None,None]
        # Reversible compression: retain outliers without letting near-constant pages dominate.
        normalized = np.arcsinh((bands-low)/scale)
        if r['sensor']=='P1':
            normalized[3:] = 0
        image = np.zeros((9,self.tile_size,self.tile_size),dtype=np.float32)
        image[:7,:height,:width] = normalized
        if r['sensor']=='Altum':
            # Raw-page ratios; not ratios of independently normalized channels.
            image[7,:height,:width] = (bands[3]-bands[2])/(bands[3]+bands[2]+1e-6)
            image[8,:height,:width] = (bands[3]-bands[4])/(bands[3]+bands[4]+1e-6)
        if self.split=='train' and random.random()<self.flip_prob:
            image,label = image[:,:,::-1].copy(),label[:,::-1].copy()
        return torch.from_numpy(image),torch.from_numpy(label.astype(np.int64))


def aligned_tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Cover an axis without padding and align the last tile to the far edge."""
    if length < tile_size:
        raise ValueError(f'Image axis {length} is smaller than tile size {tile_size}')
    if not 0 < stride <= tile_size:
        raise ValueError('Stride must lie in (0, tile_size]')
    positions = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)
    return positions


class OverlapSpectralTileDataset(SpectralTileDataset):
    """Padding-free evaluation tiles, optionally restricted to whole frames."""

    def __init__(self, root, split: str, tile_size: int, stride: int,
                 record_indices=None):
        if split == 'train':
            raise ValueError('Overlap evaluation dataset is not a training sampler')
        if not 0 < stride < tile_size:
            raise ValueError('Overlap stride must lie in (0, tile_size)')
        super().__init__(root, split, tile_size)
        self.all_records = self.records
        if record_indices is not None:
            selected = [int(index) for index in record_indices]
            self.records = [self.all_records[index] for index in selected]
        self.stride = int(stride)
        self.tiles = []
        for frame_index, record in enumerate(self.records):
            tops = aligned_tile_positions(record['height'], self.tile_size, self.stride)
            lefts = aligned_tile_positions(record['width'], self.tile_size, self.stride)
            self.tiles.extend((frame_index, left, top) for top in tops for left in lefts)
