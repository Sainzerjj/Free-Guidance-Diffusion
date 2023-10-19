from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
import torch, gc
from functools import partial
import argparse
from torch import tensor
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel, CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionAttendAndExcitePipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
    
)
from diffusers.utils import logging
from diffusers.loaders import TextualInversionLoaderMixin
from utils.ptp_utils import AttentionStore, Hook, CustomAttnProcessor, get_features
from copy import deepcopy
import copy


logger = logging.get_logger(__name__)

class StableDiffusionFreeGuidancePipeline(StableDiffusionAttendAndExcitePipeline):    
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: AutoTokenizer,
            unet: UNet2DConditionModel,
            scheduler: LMSDiscreteScheduler,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )
        print("Model loaded successfully!")
    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            final_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return final_prompt_embeds, prompt_embeds

    def check_inputs(
        self,
        prompt,
        token_indices,
        bboxes,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if token_indices is not None:
            if isinstance(token_indices, list):
                if isinstance(token_indices[0], list):
                    if isinstance(token_indices[0][0], list):
                        token_indices_batch_size = len(token_indices)
                    elif isinstance(token_indices[0][0], int):
                        token_indices_batch_size = 1
                    else:
                        raise TypeError(
                            "`token_indices` must be a list of lists of integers or a list of integers."
                        )
                else:
                    raise TypeError(
                        "`token_indices` must be a list of lists of integers or a list of integers."
                    )
            else:
                raise TypeError(
                    "`token_indices` must be a list of lists of integers or a list of integers."
                )

        if bboxes is not None:
            if isinstance(bboxes, list):
                if isinstance(bboxes[0], list):
                    if (
                        isinstance(bboxes[0][0], list)
                        and len(bboxes[0][0]) == 4
                        and all(isinstance(x, float) for x in bboxes[0][0])
                    ):
                        bboxes_batch_size = len(bboxes)
                    elif (
                        isinstance(bboxes[0], list)
                        and len(bboxes[0]) == 4
                        and all(isinstance(x, float) for x in bboxes[0])
                    ):
                        bboxes_batch_size = 1
                    else:
                        print(isinstance(bboxes[0], list), len(bboxes[0]))
                        raise TypeError(
                            "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                        )
                else:
                    print(isinstance(bboxes[0], list), len(bboxes[0]))
                    raise TypeError(
                        "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                    )
            else:
                print(isinstance(bboxes[0], list), len(bboxes[0]))
                raise TypeError(
                    "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                )

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if token_indices is not None and token_indices_batch_size != prompt_batch_size:
            raise ValueError(
                f"token indices batch size must be same as prompt batch size. token indices batch size: {token_indices_batch_size}, prompt batch size: {prompt_batch_size}"
            )

        if bboxes is not None and bboxes_batch_size != prompt_batch_size:
            raise ValueError(
                f"bbox batch size must be same as prompt batch size. bbox batch size: {bboxes_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def do_self_guidance(self, time, T, scheduler):
        if type(scheduler).__name__ == "DDPMScheduler":
            if time <= int((5*T)/16): return True
            elif time >= int(T - T/32): return False
            elif time % 2 == 0: return True
            else: return False
        if type(scheduler).__name__ == "DDIMScheduler":
            if time <= int((3*T)/16): return True
            elif time >= int(T - T/32): return False
            elif time % 2 == 0: return True
            else: return False
        elif type(scheduler).__name__ == "LMSDiscreteScheduler":
            if time <= int(T/5): return True
            elif time >= T - 5: return False
            elif time % 2 == 0: return True
            else: return False
        elif type(scheduler).__name__ == "DPMSolverMultistepScheduler":
            if time <= int(2*T/5): return True
            elif time >= T - 5: return False
            elif time % 2 == 0: return True
            else: return False
            
    
    def all_word_indexes(self, prompt, object_to_edit=None, **kwargs):
        """Extracts token indexes by treating all words in the prompt as separate objects."""
        prompt_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
        if object_to_edit is not None: 
            obj_inputs = self.tokenizer(object_to_edit, add_special_tokens=False).input_ids
            obj_idx = torch.cat([torch.where(prompt_inputs == o)[1] for o in obj_inputs])
            a = set([i for i, o in enumerate(prompt_inputs[0]) if o not in obj_inputs])
            b = set(torch.where(prompt_inputs < 49405)[1].numpy())
            other_idx = tensor(list(a&b))
            return obj_idx, other_idx
        else: return torch.where(prompt_inputs < 49405)[1]

    def choose_object_indexes(self, prompt, objects:list=None, object_to_edit=None):
        """Extracts token indexes only for user-defined objects."""
        prompt_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
        if object_to_edit is not None: 
            obj_inputs = self.tokenizer(object_to_edit, add_special_tokens=False).input_ids
            obj_idx = torch.cat([torch.where(prompt_inputs == o)[1] for o in obj_inputs])
            if object_to_edit in objects: objects.remove(object_to_edit)
        other_idx = []
        for o in objects:
            inps = self.tokenizer(o, add_special_tokens=False).input_ids
            other_idx.append(torch.cat([torch.where(prompt_inputs == o)[1] for o in inps]))
        if object_to_edit is None: return torch.cat(other_idx)
        else: return obj_idx, torch.cat(other_idx)

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = CustomAttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet)

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def prepare_attention(self, pred_type='ori', set_store=True):
        for name, module in self.unet.attn_processors.items(): 
            module.set_storage(set_store, pred_type)

    def sample(self, latents, scheduler, t, feature_layer, guidance_scale, cond_prompt_embeds, prompt_embeds, cross_attention_kwargs, hook=None, pred_type='edit', set_store=True, do_classifier_free_guidance=True):   
        latent_model_input = scheduler.scale_model_input(
            latents, t
        )
        self.prepare_attention(pred_type=pred_type, set_store=set_store)
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=cond_prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample
        feats = hook.feats if feature_layer is not None else None
        if pred_type == 'edit':
            self.unet.zero_grad()
        # perform guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = scheduler.scale_model_input(
                    latent_model_input, t
                )

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        return noise_pred, feats
    
    def decode(self, latents, output_type, device, prompt_embeds):
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)
            # 9. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(
            #     image, device, prompt_embeds.dtype
            # )
            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        return image, None




    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        ori_prompt: Union[str, List[str]] = None,
        token_indices: Union[List[List[List[int]]], List[List[int]]] = None,
        bboxes: Union[
            List[List[List[float]]],
            List[List[float]],
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_guidance_iter_per_step: int = 5,
        is_guidance: bool = True,
        guidance_func=None,
        g_weight: int =10,
        feature_layer = None, 
        objects: list = None,
        obj_to_edit = None
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            token_indices,
            bboxes,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
         # 3. Encode input prompt
        prompt_embeds, cond_prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if ori_prompt is None :
            ori_prompt = prompt
        ori_prompt_embeds, ori_cond_prompt_embeds = self._encode_prompt(
            ori_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        edit_scheduler = copy.deepcopy(self.scheduler)


        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        # ori_latents = latents
        latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        # if ori_prompt is not None:
        #     ori_latents = self.prepare_latents(
        #         batch_size * num_images_per_prompt,
        #         num_channels_latents,
        #         height,
        #         width,
        #         prompt_embeds.dtype,
        #         device,
        #         generator,
        #         ori_latents,
        #     )
        # else:
        ori_latents = latents.clone().detach()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.attention_store = AttentionStore()
        self.register_attention_control()
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # set up the hook to collect activations from feature_layer
        g_name = guidance_func.func.__name__ if isinstance(guidance_func, partial) else guidance_func.__name__
        if g_name not in ['edit_appearance'] and feature_layer is None:
            feature_layer = self.unet.up_blocks[-1].resnets[-2]
        if feature_layer is not None: hook = Hook(feature_layer, get_features)
        else: hook = None
        
        # get indexes of editable and non-editable objects from token sequence
        if self.all_word_indexes.__name__ == 'choose_object_indexes' and objects is None:
            raise ValueError('Provide a list of object strings from the prompt.')
        if g_name not in ['edit_layout', 'edit_appearance', 'edit_layout_by_feature'] and obj_to_edit is None:
            raise ValueError('Provide an object string for editing.')
        if objects is None:
            indices = self.all_word_indexes(prompt, objects=objects, object_to_edit=obj_to_edit)
        else:
            indices = self.choose_object_indexes(prompt, objects=objects, object_to_edit=obj_to_edit)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):    
                ori_noise_pred, ori_feats = self.sample(ori_latents, self.scheduler, t, feature_layer, guidance_scale, ori_cond_prompt_embeds, ori_prompt_embeds, cross_attention_kwargs, hook, pred_type='ori', set_store=True, do_classifier_free_guidance=do_classifier_free_guidance)
                ori_latents = self.scheduler.step(
                            ori_noise_pred, t, ori_latents, **extra_step_kwargs
                ).prev_sample
                if is_guidance:
                    with torch.enable_grad():
                        latents = latents.clone().detach().requires_grad_(True)
                        for guidance_iter in range(max_guidance_iter_per_step):
                            edit_noise_pred, edit_feats = self.sample(latents, edit_scheduler, t, feature_layer, guidance_scale, cond_prompt_embeds, prompt_embeds, cross_attention_kwargs, hook, pred_type='edit', set_store=True, do_classifier_free_guidance=do_classifier_free_guidance)
                            if self.do_self_guidance(i, len(self.scheduler.timesteps), self.scheduler):
                                loss = guidance_func(self.attention_store, indices, ori_feats=ori_feats, edit_feats=edit_feats)
                                grad_cond = torch.autograd.grad(
                                    loss.requires_grad_(True),
                                    [latents],
                                    retain_graph=True,
                                )[0]
                                if isinstance(self.scheduler, LMSDiscreteScheduler):
                                    sig_t = self.scheduler.sigmas[i]
                                else:
                                    sig_t = 1 - self.scheduler.alphas_cumprod[t]
                                edit_noise_pred += g_weight * sig_t * grad_cond
                        # compute the previous noisy sample x_t -> x_t-1
                        latents = edit_scheduler.step(
                            edit_noise_pred, t, latents, **extra_step_kwargs
                        ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                gc.collect()
                torch.cuda.empty_cache()
        
        edit_image, has_nsfw_concept = self.decode(latents, output_type, device, prompt_embeds)
        ori_image, ori_has_nsfw_concept = self.decode(ori_latents, output_type, device, prompt_embeds)

        
        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return [StableDiffusionPipelineOutput(
            images=edit_image, nsfw_content_detected=has_nsfw_concept
        ), StableDiffusionPipelineOutput(
            images=ori_image, nsfw_content_detected=ori_has_nsfw_concept
        )]


# if __name__ == '__main__':
#     print("Start Inference!")
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_id', type=str, default="/data/zsz/models/storage_file/models/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9")
    # parser.add_argument('--seed', type=int, default=1234)
    # # 改变shape
    # # 改变appearance
    # # 改变location
    # # 改变size
    # args = parser.parse_args()
    # # ded79e214aa69e42c24d3f5ac14b76d568679cc2
    # pipe = StableDiffusionFreeGuidancePipeline.from_pretrained(args.model_id)
    # if args.seed is None: seed = int(torch.rand((1,)) * 1000000)
    # generator=torch.manual_seed(args.seed)
    # pipe(generator=generator, mode="")
    
    