import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Type
from change_detection.sam.common import LayerNorm2d, MLPBlock
from change_detection.sam.transformer import TwoWayTransformer

class MLP(nn.Module):
    """MLP for hypernetworks and mask quality prediction."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoder(nn.Module):
    """SAM Mask Decoder for change detection, outputs only masks without prompt inputs."""
    def __init__(
        self,
        *,
        transformer_dim: int = 256,  # Match ImageEncoderViT output channels
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Args:
            transformer_dim (int): Channel dimension of the transformer.
            transformer (nn.Module): Transformer for mask prediction.
            num_multimask_outputs (int): Number of masks to predict when disambiguating masks.
            activation (nn.Module): Activation function for upscaling.
            iou_head_depth (int): Depth of the MLP for mask quality prediction.
            iou_head_hidden_dim (int): Hidden dimension of the MLP for mask quality prediction.
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),  # To 32x32
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),  # To 64x64
            activation(),
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),  # To 128x128
            activation(),
            nn.ConvTranspose2d(transformer_dim // 16, transformer_dim // 32, kernel_size=2, stride=2),  # To 256x256
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 32, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        multimask_output: bool,
    ) -> torch.Tensor:
        """
        Predict masks given image embeddings and positional encoding.

        Args:
            image_embeddings (torch.Tensor): Embeddings from the image encoder, shape [batch, 256, 16, 16].
            image_pe (torch.Tensor): Positional encoding, shape [batch, 256, 16, 16].
            multimask_output (bool): Whether to return multiple masks or a single mask.

        Returns:
            torch.Tensor: Predicted masks, shape [batch, num_mask_tokens, 256, 256] or [batch, 1, 256, 256].
        """
        masks, _ = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
        )
        # Select the correct mask(s) for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks and IoU (IoU not returned in forward)."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        tokens = output_tokens

        # Use image embeddings directly
        src = image_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions (not returned)
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

if __name__ == '__main__':
    # 测试解码器
    transformer = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)
    decoder = MaskDecoder(transformer_dim=256, transformer = transformer , num_multimask_outputs=1)
    image_embeddings = torch.randn(2, 256, 16, 16)
    image_pe = torch.randn(2, 256, 16, 16)
    masks = decoder(image_embeddings, image_pe, multimask_output=True)
    print(f"Output mask shape: {masks.shape}")  # 应为 [2, 3, 256, 256]
    print("Decoder forward pass successful.")
