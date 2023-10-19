import torch
from torch import tensor
from einops import rearrange
from .functions import normalize
import numpy as np
import fastcore.all as fc
import math
import torch.nn.functional as F
from sklearn.decomposition import PCA
import torchvision.transforms as T
from PIL import Image
import os

# the calculation of character or attribute
def threshold_attention(attn, s=10):
    norm_attn = s * (normalize(attn) - 0.5)
    return normalize(norm_attn.sigmoid())

def get_shape(attn, s=20): 
    return threshold_attention(attn, s)

def get_size(attn): 
    return 1/attn.shape[-2] * threshold_attention(attn).sum((1,2)).mean()

def get_centroid(attn):
    if not len(attn.shape) == 3: attn = attn[:,:,None]
    h = w = int(tensor(attn.shape[-2]).sqrt().item())
    hs = torch.arange(h).view(-1, 1, 1).to(attn.device)
    ws = torch.arange(w).view(1, -1, 1).to(attn.device)
    attn = rearrange(attn.mean(0), '(h w) d -> h w d', h=h)
    weighted_w = torch.sum(ws * attn, dim=[0,1])
    weighted_h = torch.sum(hs * attn, dim=[0,1])
    return torch.stack([weighted_w, weighted_h]) / attn.sum((0,1))

def get_appearance(attn, feats):
    # attn_fit = attn.permute(1, 0)
    # attn_fit = attn_fit.detach().cpu().numpy()
    # pca = PCA(n_components=3)
    # pca.fit(attn_fit)
    # feature_maps_pca = pca.transform(attn_fit)  # N X 3
    # pca_img = feature_maps_pca.reshape(1, -1, 3)  # B x (H * W) x 3
    # pca_img = pca_img.reshape(32, 32, 3)
    # pca_img_min = pca_img.min(axis=(0, 1))
    # pca_img_max = pca_img.max(axis=(0, 1))
    # pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
    # pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
    # pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
    # pca_img.save(os.path.join(f"1.png"))
    if not len(attn.shape) == 3: attn = attn[:,:,None]
    h = w = int(tensor(attn.shape[-2]).sqrt().item())
    shape = get_shape(attn).detach().mean(0).view(h,w,attn.shape[-1])
    feats = feats.mean((0,1))[:,:,None]
    return (shape * feats).sum() / shape.sum()

def get_attns(attn_storage):
    if attn_storage is not None:
        origs = attn_storage.maps('ori')
        edits = attn_storage.maps('edit')
    return origs, edits




# get character or attribute from attention layer or resnet layers
def fix_shapes_l1(orig_attns, edit_attns, indices, tau=1):
    shapes = []
    for location in ["mid", "up", "down"]:
        for o in indices:
            deltas = []
            for edit_attn_map_integrated, ori_attn_map_integrated in zip(edit_attns[location], orig_attns[location]):
                edit_attn_map = edit_attn_map_integrated.chunk(2)[1]
                ori_attn_map = ori_attn_map_integrated.chunk(2)[1]
                b, i, _ = edit_attn_map.shape
                H = W = int(np.sqrt(i))
                orig, edit = ori_attn_map[:,:,o], edit_attn_map[:,:,o]
                delta = tau * get_shape(orig) - get_shape(edit)
                deltas.append(delta.mean())
            shapes.append(torch.stack(deltas).mean())
    return torch.stack(shapes).mean()

def fix_shapes_l2(orig_attns, edit_attns, indices, tau=fc.noop):
    shapes = []
    for location in ["mid", "up", "down"]:
        for o in indices:
            deltas = []
            for edit_attn_map_integrated, ori_attn_map_integrated in zip(edit_attns[location], orig_attns[location]):
                edit_attn_map = edit_attn_map_integrated.chunk(2)[1]
                ori_attn_map = ori_attn_map_integrated.chunk(2)[1]
                orig, edit = ori_attn_map[:,:,o], edit_attn_map[:,:,o]
                if len(orig.shape) < 3: orig, edit = orig[...,None], edit[...,None]
                delta = (tau(get_shape(orig)) - get_shape(edit)).pow(2).mean()
                deltas.append(delta.mean())
            shapes.append(torch.stack(deltas).mean())
    return torch.stack(shapes).mean()

def fix_shapes_l3(orig_attns, edit_attns, indices, tau=fc.noop):
    shapes = []
    for location in ["mid", "up", "down"]:
        for o in indices:
            deltas = []
            for edit_attn_map_integrated, ori_attn_map_integrated in zip(edit_attns[location], orig_attns[location]):
                edit_attn_map = edit_attn_map_integrated.chunk(2)[1]
                ori_attn_map = ori_attn_map_integrated.chunk(2)[1]
                orig, edit = ori_attn_map[:,:,o], edit_attn_map[:,:,o]
                if len(orig.shape) < 3: orig, edit = orig[...,None], edit[...,None]
                t = orig + (orig.max())
                delta = (get_shape((orig + t).clip(min=0))) - get_shape(edit)
                deltas.append(delta.mean())
            shapes.append(torch.stack(deltas).mean())
    return torch.stack(shapes).mean()

def fix_appearances(orig_attns, ori_feats, edit_attns, edit_feats, indices):
    appearances = []
    for o in indices:
        orig = torch.stack([a.chunk(2)[1][:,:,o] for a in edit_attns['up'][-3:]]).mean(0)
        edit = torch.stack([b.chunk(2)[1][:,:,o] for b in orig_attns['up'][-3:]]).mean(0)
        appearances.append((get_appearance(orig, ori_feats) - get_appearance(edit, edit_feats)).pow(2).mean())
    return torch.stack(appearances).mean()

def fix_sizes(orig_attns, edit_attns, indices, tau=1):
    sizes = []
    for location in ["mid", "up", "down"]:
        for o in indices:
            for edit_attn_map_integrated, ori_attn_map_integrated in zip(edit_attns[location], orig_attns[location]):
                edit_attn_map = edit_attn_map_integrated.chunk(2)[1]
                ori_attn_map = ori_attn_map_integrated.chunk(2)[1]
                orig, edit = ori_attn_map[:,:,o], edit_attn_map[:,:,o]
                sizes.append(tau*get_size(orig) - get_size(edit))
    return torch.stack(sizes).mean()

def position_deltas(orig_attns, edit_attns, indices, target_centroid=None):
    positions = []
    for location in ["mid", "up", "down"]:
        for o in indices:
            for edit_attn_map_integrated, ori_attn_map_integrated in zip(edit_attns[location], orig_attns[location]):
                edit_attn_map = edit_attn_map_integrated.chunk(2)[1]
                ori_attn_map = ori_attn_map_integrated.chunk(2)[1]
                orig, edit = ori_attn_map[:,:,o], edit_attn_map[:,:,o]
                target = tensor(target_centroid) if target_centroid is not None else get_centroid(orig)
                positions.append(target.to(orig.device) - get_centroid(edit))
    return torch.stack(positions).mean()

def roll_shape(x, direction='up', factor=0.5):
    h = w = int(math.sqrt(x.shape[-2]))
    mag = (0,0)
    if direction == 'up': mag = (int(-h*factor),0)
    elif direction == 'down': mag = (int(-h*factor),0)
    elif direction == 'right': mag = (0,int(w*factor))
    elif direction == 'left': mag = (0,int(-w*factor))
    shape = (x.shape[0], h, h, x.shape[-1])
    x = x.view(shape)
    move = x.roll(mag, dims=(1,2))
    return move.view(x.shape[0], h*h, x.shape[-1])

def enlarge(x, scale_factor=1):
    assert scale_factor >= 1
    h = w = int(math.sqrt(x.shape[-2]))
    x = rearrange(x, 'n (h w) d -> n d h w', h=h)
    x = F.interpolate(x, scale_factor=scale_factor)
    new_h = new_w = x.shape[-1]
    x_l, x_r = (new_w//2) - w//2, (new_w//2) + w//2
    x_t, x_b = (new_h//2) - h//2, (new_h//2) + h//2
    x = x[:,:,x_t:x_b,x_l:x_r]
    return rearrange(x, 'n d h w -> n (h w) d', h=h) * scale_factor

def shrink(x, scale_factor=1):
    assert scale_factor <= 1
    h = w = int(math.sqrt(x.shape[-2]))
    x = rearrange(x, 'n (h w) d -> n d h w', h=h)
    sf = int(1/scale_factor)
    new_h, new_w = h*sf, w*sf
    x1 = torch.zeros(x.shape[0], x.shape[1], new_h, new_w).to(x.device)
    x_l, x_r = (new_w//2) - w//2, (new_w//2) + w//2
    x_t, x_b = (new_h//2) - h//2, (new_h//2) + h//2
    x1[:,:,x_t:x_b,x_l:x_r] = x
    shrink = F.interpolate(x1, scale_factor=scale_factor)
    return rearrange(shrink, 'n d h w -> n (h w) d', h=h) * scale_factor

def resize(x, scale_factor=1):
    if scale_factor > 1: return enlarge(x)
    elif scale_factor < 1: return shrink(x)
    else: return x




# guidance functions
def edit_layout(attn_storage, indices, appearance_weight=0.5, ori_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    return appearance_weight * fix_appearances(origs, ori_feats, edits, edit_feats, indices, **kwargs)

def edit_appearance(attn_storage, indices, shape_weight=1, **kwargs):
    origs, edits = get_attns(attn_storage)
    return shape_weight * fix_shapes_l1(origs, edits, indices)

def resize_object_by_size(attn_storage, indices, relative_size=2, shape_weight=1, size_weight=1, appearance_weight=0.1, ori_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    if len(indices) > 1: 
        obj_idx, other_idx = indices
        indices = torch.cat([obj_idx, other_idx])
    shape_term = shape_weight * fix_shapes_l1(origs, edits, indices)
    appearance_term = appearance_weight * fix_appearances(origs, ori_feats, edits, edit_feats, indices)
    size_term = size_weight * fix_sizes(origs, edits, indices, tau=relative_size)
    return shape_term + appearance_term + size_term

def resize_object_by_shape(attn_storage, indices, tau=fc.noop, shape_weight=1, size_weight=1, appearance_weight=0.1, ori_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    # orig_selfs = [v['orig'] for k,v in attn_storage.storage.items() if 'attn1' in k][-1]
    # edit_selfs = [v['edit'] for k,v in attn_storage.storage.items() if 'attn1' in k][-1]
    if len(indices) > 1:
        obj_idx, other_idx = indices
        indices = torch.cat([obj_idx, other_idx])
    shape_term = shape_weight * fix_shapes_l1(origs, edits, other_idx)
    appearance_term = appearance_weight * fix_appearances(origs, ori_feats, edits, edit_feats, indices)
    size_term = size_weight * fix_shapes_l3(origs, edits, obj_idx, tau=tau)
    # self_term = self_weight*fix_selfs(orig_selfs, edit_selfs)
    return shape_term + appearance_term + size_term

def move_object_by_centroid(attn_storage, indices, target_centroid=None, shape_weight=1, size_weight=1, appearance_weight=0.5, position_weight=1, ori_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    if len(indices) > 1: 
        obj_idx, other_idx = indices
        indices = torch.cat([obj_idx, other_idx])
    shape_term = shape_weight * fix_shapes_l1(origs, edits, indices)
    appearance_term = appearance_weight * fix_appearances(origs, ori_feats, edits, edit_feats, indices)
    size_term = size_weight * fix_sizes(origs, edits, obj_idx)
    position_term = position_weight * position_deltas(origs, edits, obj_idx, target_centroid=target_centroid)
    return shape_term + appearance_term + size_term + position_term

def move_object_by_shape(attn_storage, indices, tau=fc.noop, shape_weight=1, appearance_weight=0.5, position_weight=1, ori_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    # orig_selfs = [v['orig'] for k,v in attn_storage.storage.items() if 'attn1' in k and v['orig'].shape[-1] == 4096]
    # edit_selfs = [v['edit'] for k,v in attn_storage.storage.items() if 'attn1' in k and v['orig'].shape[-1] == 4096]
    if len(indices) > 1: 
        obj_idx, other_idx = indices
        indices = torch.cat([obj_idx, other_idx])
    shape_term = shape_weight * fix_shapes_l1(origs, edits, other_idx)
    appearance_term = appearance_weight * fix_appearances_by_feature(ori_feats, edit_feats, indices)
    # size_term = size_weight*fix_sizes(origs, edits, obj_idx)
    # position_term = position_weight*position_deltas_2(origs, edits, obj_idx, target_centroid=target_centroid)
    # self_term = self_weight*fix_selfs_2(orig_selfs, edit_selfs, t=t)
    move_term = position_weight * fix_shapes_l2(origs, edits, obj_idx, tau=tau)
    return move_term + shape_term + appearance_term

def fix_appearances_by_feature(ori_feats, edit_feats, indices):
    appearances = []
    for o in indices: appearances.append((ori_feats - edit_feats).pow(2).mean())
    return torch.stack(appearances).mean()

def edit_layout_by_feature(attn_storage, indices, appearance_weight=0.5, ori_feats=None, edit_feats=None, **kwargs):
    return appearance_weight * fix_appearances_by_feature(ori_feats, edit_feats, indices)