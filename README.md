# Free-Guidance-Diffusion
A method that provides greater control over generated images by guiding the internal representations of the pre-trained Stable Diffusion (SDv1.5). This idea is inspired by [Diffusion Self-Guidance for Controllable Image Generation](https://arxiv.org/pdf/2306.00986.pdf) (NIPS 2023).

The method allows for various modifications, including changing the position or size of specific objects, combining the appearance of an object from one image with the layout of another image, and merging objects from multiple images into a single image. 

The core implementation is **StableDiffusionFreeGuidancePipeline** written based on [🧨diffusers](https://huggingface.co/docs/diffusers/index), which is defined in   `free_guidance.py`. The class inherits from the [StableDiffusionAttendAndExcitePipeline](https://huggingface.co/spaces/AttendAndExcite/Attend-and-Excite) and can be easily used. The file `experiments.ipynb` provides some visualization attempts as a reference for improvement. All guidance functions are located in `./utils/guidance_function.py`. All visualization methods are defined in `./utils/vis_utils.py`.

The biggest challenege is the weights are very sensitive and the method performs worse as the prompts get more complex — subjects of the image interact, and it becomes harder to isolate the attention of specific tokens.


