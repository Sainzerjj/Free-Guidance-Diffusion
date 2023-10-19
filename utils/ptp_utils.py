from diffusers.models.attention_processor import AttnProcessor, Attention
import torch
from functools import partial
import fastcore.all as fc
import numpy as np
from typing import List
from PIL import Image
from .vis_utils import preprocess_image

def get_features(hook, layer, inp, out):
    if not hasattr(hook, 'feats'): hook.feats = out
    hook.feats = out

def get_latents_from_image(pipe, img_path, device):
    if img_path is None: return None
    img = Image.open(img_path)
    img = img.convert('RGB')
    image = preprocess_image(img).to(device)
    init_latents = pipe.vae.encode(image.half()).latent_dist.sample() * 0.18215
    shape = init_latents.shape
    noise = torch.randn(shape, device=device)
    timesteps = pipe.scheduler.timesteps[0] 
    timesteps = torch.tensor([timesteps], device=device)
    init_latents = pipe.scheduler.add_noise(init_latents, noise, timesteps).half()
    return init_latents

class Hook():
    def __init__(self, model, func): self.hook = model.register_forward_hook(partial(func, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()


class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {'ori' : {"down": [], "mid": [], "up": []}, 'edit' : {"down": [], "mid": [], "up": []}}
    def __init__(self, attn_res=[4096, 1024, 256, 64]): 
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res

    def __call__(self, attention_map, is_cross, place_in_unet: str, pred_type='ori'): 
        # if not name in self.step_store: 
        #     self.step_store[name] = {}
        # self.step_store[name][pred_type] = attention_map
        if self.cur_att_layer >= 0 and is_cross:
            if attention_map.shape[1] in self.attn_res:
                self.step_store[pred_type][place_in_unet].append(attention_map)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps(pred_type)
    
    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    
    def maps(self, block_type: str):
        return self.attention_store[block_type]

    def between_steps(self, pred_type='ori'):
        self.attention_store[pred_type] = self.step_store[pred_type]
        self.step_store = self.get_empty_store()



class CustomAttnProcessor(AttnProcessor):
    def __init__(self, attnstore, place_in_unet=None): 
        super().__init__()
        fc.store_attr()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.store = False

    def set_storage(self, store, pred_type): 
        self.store = store
        self.pred_type = pred_type

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
     
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        if self.store: 
            self.attnstore(attention_probs, is_cross, self.place_in_unet, pred_type=self.pred_type) ## stores the attention maps in attn_storage
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states

