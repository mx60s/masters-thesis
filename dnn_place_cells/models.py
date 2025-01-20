import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import tensorflow as tf
from tensorflow.keras.models import Model

"""
Model loading function and definitions.
"""

def load_model(model_name, output_layer=None, input_shape=(224, 224, 3)):
    if 'simclr' in model_name:
        if model_name == 'simclrv2_r50_1x_sk0':
            model_path = f'model_zoo/{model_name}/saved_model'
            model = _build_simclr(model_path, output_layer)
            from model_zoo.simclrv2_r50_1x_sk0 import preprocessing
            preprocess_func = preprocessing.preprocess_func
    else:
        if 'vit' in model_name:
            if model_name == 'vit_b16':
                from transformers import AutoImageProcessor, TFViTModel
                model = TFViTModel.from_pretrained(
                    'google/vit-base-patch16-224-in21k',
                    cache_dir='model_zoo/vit_b16'
                )
                preprocess_func = AutoImageProcessor.from_pretrained(
                    "google/vit-base-patch16-224-in21k",
                    cache_dir='model_zoo/vit_b16'
                )
            elif model_name == 'vit_b16_untrained':
                from transformers import AutoImageProcessor, ViTConfig, TFViTModel                        
                config = ViTConfig()
                model = TFViTModel(config)
                preprocess_func = AutoImageProcessor.from_pretrained(
                    "google/vit-base-patch16-224-in21k",
                    cache_dir='model_zoo/vit_b16'
                )

        else:
            if model_name == 'vgg16':
                model = tf.keras.applications.VGG16(
                    weights='imagenet', 
                    include_top=True, 
                    input_shape=input_shape,
                    classifier_activation=None
                )
                preprocess_func = tf.keras.applications.vgg16.preprocess_input

            elif model_name == 'vgg16_untrained':
                model = tf.keras.applications.VGG16(
                    weights=None, 
                    include_top=True, 
                    input_shape=input_shape,
                    classifier_activation=None
                )
                preprocess_func = tf.keras.applications.vgg16.preprocess_input

            elif model_name == 'resnet50':
                model = tf.keras.applications.ResNet50(
                    weights='imagenet', 
                    include_top=True, 
                    input_shape=input_shape,
                    classifier_activation=None
                )
                preprocess_func = tf.keras.applications.resnet50.preprocess_input

            elif model_name == 'resnet50_untrained':
                model = tf.keras.applications.ResNet50(
                    weights=None, 
                    include_top=True, 
                    input_shape=input_shape,
                    classifier_activation=None
                )
                preprocess_func = tf.keras.applications.resnet50.preprocess_input

            if output_layer is None:
                output_layer = model.layers[-1].name
            model = Model(inputs=model.input, outputs=model.get_layer(output_layer).output)

    return model, preprocess_func


def _build_simclr(model_path, output_layer):
    class SimCLRv2(tf.keras.Model):
        def __init__(self):
            super(SimCLRv2, self).__init__()
            self.saved_model = \
                tf.saved_model.load(model_path)
            self.output_layer = output_layer

        def call(self, inputs):
            # print((self.saved_model(
            #         inputs, trainable=False)).keys())
            return \
                self.saved_model(
                    inputs, trainable=False)[self.output_layer]
    return SimCLRv2()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model, preprocess_func = load_model(model_name='vit_b16')

    print(
    # model.vit._layers[1]                                              # is TFViTEncoder, 
    # model.vit._layers[1]._layers                                      # contain 12 TFViTLayer
    # model.vit._layers[1]._layers[0][0])                               # 1/12 TFViTLayer (Transformer block)
    # model.vit._layers[1]._layers[0][0]._layers,                       # has Attention, layernorm, etc.
    # model.vit._layers[1]._layers[0][0]._layers[0]._layers,            # TFViTSelfAttention+FViTSelfOutput
    model.vit._layers[1]._layers[0][0]._layers[0]._layers[0]._layers,   # 3 Dense layers and dropout (Q,K,V)
    )


'''
import torch.nn.functional as F

def get_predcode_output_by_layer(model, features, layer_name):
    x = model.encoder.conv1(features)
    x = model.encoder.bn1(x)
    x = F.relu(x)
    x = model.encoder.maxpool(x)
    x = model.encoder.down1(x)

    if layer_name == 'block2_pool':
        return x

    x = model.encoder.down2(x)

    if layer_name == 'post_pred':
        x = torch.unsqueeze(x, 1) # sequence length
        residual = x
        x = model.attention_1(x, x, x, model.mask)[0]
        x = x + residual
        x = F.layer_norm(x, x.shape[1:])

        residual = x
        x = model.ffn_1(x) + residual
        x = F.layer_norm(x, x.shape[1:])

        residual = x
        x = model.attention_2(x, x, x, model.mask)[0]
        x = x + residual
        x = F.layer_norm(x, x.shape[1:])

        residual = x
        x = model.ffn_2(x) + residual
        x = F.layer_norm(x, x.shape[1:])

        x = torch.squeeze(x)
    
    return x
    

    import torch

# Model selection

model_name = 'predictive-coding-small-world-new'

model = PredictiveCoder(
    in_channels=3,
    out_channels=3,
    layers=[2, 2, 2, 2],
    seq_len=20,
)
    
model.load_state_dict(torch.load(f'/home/mag/predictive-coding-recovers-maps/notebooks/experiments/{model_name}/best.ckpt'))

model.eval()
model = model.to('cuda:1')

# Layer selection 
layer_names = ['block2_pool', 'post_pred']
env = 'small_world_r17'

get_outputs_fn = get_predcode_output_by_layer

from glob import glob
import re
import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor, Normalize

# Set up dataset
len_ims = len(glob(f'data/minecraft/{env}/*'))
images = []

for i in range(len_ims):
    image = Image.open(f'data/minecraft/{env}/{i}.png')
    image = Normalize([121.6697/255, 149.3242/255, 154.9510/255], [40.7521/255,  47.7267/255, 103.2739/255])(ToTensor()(image))

    images.append(image)


positions = np.load(f'data/minecraft/{env}_pos.npy')
images = torch.stack(images, dim=0)

if env == 'biggest_world_r17':
    x_min = -20
    x_max = 20
    y_min = -30
    y_max = 35
    multiplier = 1
else:
    x_min = 112
    y_min = 197
    x_max = 135
    y_max = 221
    multiplier=1


for layer_name in layer_names:
    print(f"Processing {layer_name}")
    latents = []
    locations = []
        
    bsz = 1
    for idx in range(len(images)):
        batch = images[bsz*idx:bsz*(idx+1)].to('cuda:1')
        batch_positions = positions[bsz*idx:bsz*(idx+1)]

        with torch.no_grad():
            features = get_predcode_output_by_layer(model, batch, layer_name)
            features = features.reshape(bsz, -1)
            latents.append(features.cpu())
    
        locations.append(batch_positions)
                    
    model_reps = torch.cat(latents, dim=0)

    f = f'results/{env}/2d/uniform/{model_name}/{layer_name}/'
    os.makedirs(f, exist_ok=True)
    
    np.save(f'{f}/model_reps.npy', model_reps)
    


'''
