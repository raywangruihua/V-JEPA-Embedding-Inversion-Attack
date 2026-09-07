import torch
import umap
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import normalize


def normalise_vjepa(embeddings: torch.Tensor) -> np.ndarray:
    """ Pool spatiotemporal patches, normalize and convert a batch of VJEPA
    embeddings into numpy arrays for downstream processing.

    Expects a batch of embeddings with shape (B, 1, N, D):
    - B = Number of embeddings
    - N = Number of spatiotemporal patches, the number of smaller 3D tensors
    the video is split into along temporal and spatial axes
    - D = Embedding dimension (or number of features), depends on ViT backbone
    of model

    Normalisation makes a few assumptions:
    - Each embedding was encoded from 1 video, i.e. the 2nd dimension is 1
    - ViT-L backbone is used, which has 1024 features, i.e. D = 1024
    - Spatiotemporal patch has shape (8, 14, 14) which is used to train the
      official meta V-JEPA models
    """

    return normalize(embeddings.permute(0, 1, 3, 2)
                               .reshape(-1, 1024, 8, 14, 14)
                               .mean(dim=(2, 3, 4))
                               .cpu().numpy(), norm="l2")


def normalise_ltx2_vae(embeddings: torch.Tensor) -> np.ndarray:
    """ Pool temporal and spatial axes, normalize and convert a batch of
    LTX2 VAE embeddings into numpy arrays for downstream processing.

    Expects a batch of embeddings with shape (B, 1, C=128, T, H, W):
    - B = Number of embeddings
    - C = Number of channels
    - T = (Number of video frames) / 2
    - H = (Height of video in pixels) / 32
    - W = (Width of video in pixels) / 32

    Assumes that each embedding was encoded from 1 video, i.e. the 2nd
    dimension is 1.
    """
    return normalize(embeddings.squeeze(1)
                               .mean(dim=(2, 3, 4))
                               .cpu().numpy(), norm="l2")


def plot_umap(
    model_embeddings: list[torch.Tensor],
    model_labels: list[str],
    model_names: list[str]
) -> None:
    """ Plots embeddings from different models in 3D space using UMAP. 
    
    Embeddings should be normalised before plotting.
    """
    model_coords = []
    for embeddings in model_embeddings:
        model_coords.append(umap.UMAP(
            n_components=3,
            metric="cosine",
            random_state=42
            ).fit_transform(embeddings))

    model_df = []
    for i, coords in enumerate(model_coords):
        model_df.append(pd.DataFrame({
            "x": coords[:,0],
            "y": coords[:,1],
            "z": coords[:,2],
            "label": model_labels[i],
            "model": model_names[i]
        }))
    df = pd.concat(model_df, ignore_index=True)

    fig = px.scatter_3d(
        df, x="x", y="y", z="z", color="model", hover_name="label")
    fig.update_traces(marker=dict(size=1))
    fig.show()
