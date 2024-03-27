import torch
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures

extra_features = DummyExtraFeatures()
domain_features = DummyExtraFeatures()
visualization_tools = None
# Load the model
model_path = '/Users/chanithas/Desktop/fyp/Procedural-Material-Graph-Generation-by-text-prompt/outputs/2024-03-26/03-43-16-graph-tf-model/checkpoints/graph-tf-model/last-v1.ckpt'

model_kwargs = {'dataset_infos': dataset_infos, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

model = DiscreteDenoisingDiffusion.load_from_checkpoint(checkpoint_path=model_path, )
# y = torch.tensor([...], dtype=torch.float)
y = torch.tensor([], size=(1, 0))
generated_graphs = model.generate_from_y(y=y, batch_size=10)