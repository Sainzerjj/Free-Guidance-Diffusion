# Free-Guidance-Diffusion
A method that provides greater control over generated images by guiding the internal representations of the pre-trained Stable Diffusion.

The method allows for various modifications, such as changing the position or size of specific objects, combining the appearance of an object from one image with the layout of another image, and merging objects from multiple images into a single image.
This idea is inspired by [Diffusion Self-Guidance for Controllable Image Generation (NIPS 2023)](https://arxiv.org/pdf/2306.00986.pdf).

The core implementation is **StableDiffusionFreeGuidancePipeline** written based on [diffusers](https://huggingface.co/docs/diffusers/index). The class is defined in **free_guidance.py** which inherits from the [StableDiffusionAttendAndExcitePipeline](https://huggingface.co/spaces/AttendAndExcite/Attend-and-Excite). The file **experiments.ipynb** provides some visualization attempts as a reference for improvement.
