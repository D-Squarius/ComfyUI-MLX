import folder_paths
import comfy.utils
import comfy.model_management
import torch

from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_audio import VAEEncodeAudio

class LTXVAudioVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        ckpt_names = folder_paths.get_filename_list("checkpoints")
        try:
            from comfy.backends.mlx_ltx_backend import MLX_LTX_AUDIO_VAE_ALIASES, list_mlx_ltx_checkpoint_choices

            ckpt_names = sorted(set(ckpt_names).union(list_mlx_ltx_checkpoint_choices(aliases=False)).union(MLX_LTX_AUDIO_VAE_ALIASES))
        except ImportError:
            pass
        return io.Schema(
            node_id="LTXVAudioVAELoader",
            display_name="Load LTXV Audio VAE",
            category="loaders",
            inputs=[
                io.Combo.Input(
                    "ckpt_name",
                    options=ckpt_names,
                    tooltip="Audio VAE checkpoint to load.",
                )
            ],
            outputs=[io.Vae.Output(display_name="Audio VAE")],
        )

    @classmethod
    def execute(cls, ckpt_name: str) -> io.NodeOutput:
        try:
            from comfy.backends.mlx_ltx_backend import MLX_LTX_AUDIO_VAE_ALIASES, load_mlx_ltx_checkpoint, resolve_mlx_ltx_checkpoint_path

            ckpt_path = resolve_mlx_ltx_checkpoint_path(ckpt_name, alias_map=MLX_LTX_AUDIO_VAE_ALIASES)
            if ckpt_path is not None:
                return io.NodeOutput(load_mlx_ltx_checkpoint(ckpt_path)[2])
        except ImportError:
            pass

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
        sd = comfy.utils.state_dict_prefix_replace(sd, {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()

        return io.NodeOutput(vae)


class LTXVAudioVAEEncode(VAEEncodeAudio):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVAudioVAEEncode",
            display_name="LTXV Audio VAE Encode",
            category="latent/audio",
            inputs=[
                io.Audio.Input("audio", tooltip="The audio to be encoded."),
                io.Vae.Input(
                    id="audio_vae",
                    display_name="Audio VAE",
                    tooltip="The Audio VAE model to use for encoding.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="Audio Latent")],
        )

    @classmethod
    def execute(cls, audio, audio_vae) -> io.NodeOutput:
        return super().execute(audio_vae, audio)


class LTXVAudioVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVAudioVAEDecode",
            display_name="LTXV Audio VAE Decode",
            category="latent/audio",
            inputs=[
                io.Latent.Input("samples", tooltip="The latent to be decoded."),
                io.Vae.Input(
                    id="audio_vae",
                    display_name="Audio VAE",
                    tooltip="The Audio VAE model used for decoding the latent.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="Audio")],
        )

    @classmethod
    def execute(cls, samples, audio_vae) -> io.NodeOutput:
        try:
            from comfy.backends.mlx_ltx_backend import (
                is_mlx_ltx_media_latent,
                make_mlx_ltx_audio_proxy,
                mlx_ltx_media_components,
                mlx_ltx_media_passthrough_enabled,
            )

            if is_mlx_ltx_media_latent(samples):
                if mlx_ltx_media_passthrough_enabled(samples):
                    return io.NodeOutput(make_mlx_ltx_audio_proxy(samples))
                return io.NodeOutput(mlx_ltx_media_components(samples).audio)
        except ImportError:
            pass

        audio_latent = samples["samples"]
        if audio_latent.is_nested:
            audio_latent = audio_latent.unbind()[-1]
        audio = audio_vae.decode(audio_latent).movedim(-1, 1).to(audio_latent.device)
        output_audio_sample_rate = audio_vae.first_stage_model.output_sample_rate
        return io.NodeOutput(
            {
                "waveform": audio,
                "sample_rate": int(output_audio_sample_rate),
            }
        )


class LTXVEmptyLatentAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVEmptyLatentAudio",
            display_name="LTXV Empty Latent Audio",
            category="latent/audio",
            inputs=[
                io.Int.Input(
                    "frames_number",
                    default=97,
                    min=1,
                    max=1000,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Number of frames.",
                ),
                io.Int.Input(
                    "frame_rate",
                    default=25,
                    min=1,
                    max=1000,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Number of frames per second.",
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=4096,
                    display_mode=io.NumberDisplay.number,
                    tooltip="The number of latent audio samples in the batch.",
                ),
                io.Vae.Input(
                    id="audio_vae",
                    display_name="Audio VAE",
                    tooltip="The Audio VAE model to get configuration from.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="Latent")],
        )

    @classmethod
    def execute(
        cls,
        frames_number: int,
        frame_rate: int,
        batch_size: int,
        audio_vae,
    ) -> io.NodeOutput:
        """Generate empty audio latents matching the reference pipeline structure."""

        assert audio_vae is not None, "Audio VAE model is required"
        try:
            from comfy.backends.mlx_ltx_backend import is_mlx_ltx_vae

            if is_mlx_ltx_vae(audio_vae):
                num_audio_latents = audio_vae.first_stage_model.num_of_latents_from_frames(frames_number, frame_rate)
                audio_latents = torch.zeros(
                    (batch_size, 8, num_audio_latents, 16),
                    device=comfy.model_management.intermediate_device(),
                )
                return io.NodeOutput(
                    {
                        "samples": audio_latents,
                        "type": "audio",
                    }
                )
        except ImportError:
            pass

        z_channels = audio_vae.latent_channels
        audio_freq = audio_vae.first_stage_model.latent_frequency_bins

        num_audio_latents = audio_vae.first_stage_model.num_of_latents_from_frames(frames_number, frame_rate)

        audio_latents = torch.zeros(
            (batch_size, z_channels, num_audio_latents, audio_freq),
            device=comfy.model_management.intermediate_device(),
        )

        return io.NodeOutput(
            {
                "samples": audio_latents,
                "type": "audio",
            }
        )


class LTXAVTextEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        text_encoders = folder_paths.get_filename_list("text_encoders")
        ckpt_names = folder_paths.get_filename_list("checkpoints")
        try:
            from comfy.backends.mlx_ltx_backend import MLX_LTX_TEXT_ENCODER_ALIASES, list_mlx_ltx_checkpoint_choices

            text_encoders = sorted(set(text_encoders).union(MLX_LTX_TEXT_ENCODER_ALIASES))
            ckpt_names = sorted(set(ckpt_names).union(list_mlx_ltx_checkpoint_choices()))
        except ImportError:
            pass
        return io.Schema(
            node_id="LTXAVTextEncoderLoader",
            display_name="LTXV Audio Text Encoder Loader",
            category="advanced/loaders",
            description="[Recipes]\n\nltxav: gemma 3 12B",
            inputs=[
                io.Combo.Input(
                    "text_encoder",
                    options=text_encoders,
                ),
                io.Combo.Input(
                    "ckpt_name",
                    options=ckpt_names,
                ),
                io.Combo.Input(
                    "device",
                    options=["default", "cpu"],
                    advanced=True,
                )
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, text_encoder, ckpt_name, device="default"):
        clip_type = comfy.sd.CLIPType.LTXV

        try:
            from comfy.backends.mlx_ltx_backend import MLX_LTX_TEXT_ENCODER_ALIASES, load_mlx_ltx_checkpoint, resolve_mlx_ltx_checkpoint_path

            ckpt_path = resolve_mlx_ltx_checkpoint_path(ckpt_name)
            if ckpt_path is None:
                ckpt_path = resolve_mlx_ltx_checkpoint_path(text_encoder, alias_map=MLX_LTX_TEXT_ENCODER_ALIASES)
            if ckpt_path is not None:
                return io.NodeOutput(load_mlx_ltx_checkpoint(ckpt_path)[1])
        except ImportError:
            pass

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        clip_path2 = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        return io.NodeOutput(clip)


class LTXVAudioExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTXVAudioVAELoader,
            LTXVAudioVAEEncode,
            LTXVAudioVAEDecode,
            LTXVEmptyLatentAudio,
            LTXAVTextEncoderLoader,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return LTXVAudioExtension()
