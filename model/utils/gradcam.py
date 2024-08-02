# import cv2
# import numpy as np
# import torch
# import torch.nn.parallel

# class FeatureExtractor():
#     """ Class for extracting activations and
#     registering gradients from targetted intermediate layers """

#     def __init__(self, model, target_layers):
#         self.model = model
#         self.target_layers = target_layers
#         self.gradients = []

#     def save_gradient(self, grad):
#         self.gradients.append(grad)

#     def __call__(self, x):
#         outputs = []
#         self.gradients = []
#         for name, module in self.model._modules.items():
#             x = module(x)
#             if name in self.target_layers:
#                 x.register_hook(self.save_gradient)
#                 outputs += [x]
#         return outputs, x


# class ModelOutputs():
#     """ Class for making a forward pass, and getting:
#     1. The network output.
#     2. Activations from intermeddiate targetted layers.
#     3. Gradients from intermeddiate targetted layers. """

#     def __init__(self, model, feature_module, target_layers):
#         self.model = model
#         self.feature_module = feature_module
#         self.feature_extractor = FeatureExtractor(
#             self.feature_module, target_layers)

#     def get_gradients(self):
#         return self.feature_extractor.gradients

#     def __call__(self, x):
#         target_activations = []
#         for name, module in self.model._modules.items():
#             if module == self.feature_module:
#                 target_activations, x = self.feature_extractor(x)
#             elif "avgpool" in name.lower():
#                 x = module(x)
#                 x = x.view(x.size(0), -1)
#             elif "imp" in name.lower():
#                 continue
#             else:
#                 x = module(x)

#         return target_activations, x


# class GradCam:
#     def __init__(self, model, feature_module, target_layer_names, use_cuda):
#         self.model = model
#         self.feature_module = feature_module
#         # self.model.eval()
#         self.cuda = use_cuda
#         if self.cuda:
#             self.model = model.cuda()

#         self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

#     def forward(self, input):
#         return self.model(input)

#     def get_attention_map(self, input, index=None, norm=None):
#         if self.cuda:
#             features, output = self.extractor(input.cuda())
#         else:
#             features, output = self.extractor(input)

#         if index == None:
#             index = np.argmax(output.cpu().data.numpy())

#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0][index] = 1
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         if self.cuda:
#             one_hot = torch.sum(one_hot.cuda() * output)
#         else:
#             one_hot = torch.sum(one_hot * output)

#         self.feature_module.zero_grad()
#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)

#         grads_val = self.extractor.get_gradients()[-1]

#         target = features[-1].squeeze()

#         weights = torch.mean(grads_val, axis=(2, 3)).squeeze()

#         if self.cuda:
#             cam = torch.zeros(target.shape[1:]).cuda()
#         else:
#             cam = torch.zeros(target.shape[1:])

#         for i, w in enumerate(weights):
#             cam += w * target[i, :, :]

#         if norm == 'ReLU':
#             cam = torch.relu(cam)
#             cam = cam / (torch.max(cam) + 1e-6)
#         elif norm == 'Sigmoid':
#             cam = torch.sigmoid(cam)
#         else:
#             cam = cam - torch.min(cam)
#             cam = cam / (torch.max(cam) + 1e-6)

#         return cam, output

#     def __call__(self, input, index=None, norm='ReLU'):
#         if self.cuda:
#             features, output = self.extractor(input.cuda())
#         else:
#             features, output = self.extractor(input)

#         if index == None:
#             index = np.argmax(output.cpu().data.numpy())

#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0][index] = 1
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         if self.cuda:
#             one_hot = torch.sum(one_hot.cuda() * output)
#         else:
#             one_hot = torch.sum(one_hot * output)

#         self.feature_module.zero_grad()
#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)

#         grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

#         target = features[-1]
#         target = target.cpu().data.numpy()[0, :]

#         weights = np.mean(grads_val, axis=(2, 3))[0, :]
#         cam = np.zeros(target.shape[1:], dtype=np.float32)

#         for i, w in enumerate(weights):
#             cam += w * target[i, :, :]

#         # use when visualizing explanation
#         # cam = cam - np.min(cam)
#         # cam = cv2.resize(cam, input.shape[2:])
#         # if norm:
#         #     cam = cam / (np.max(cam) + 1e-6)

#         # new way of nomralization ReLU(x) / (max(ReLU(x)) + 1e-6)
#         if norm == 'ReLU':
#             zeros = np.zeros(cam.shape, dtype=cam.dtype)
#             cam = np.maximum(cam, zeros)
#             cam = cv2.resize(cam, input.shape[2:])
#             cam = cam / (np.max(cam) + 1e-6)
#         else:
#             cam = cam - np.min(cam)
#             cam = cv2.resize(cam, input.shape[2:])
#             cam = cam / (np.max(cam) + 1e-6)

#         # print('final cam.shape:', cam.shape)
#         return cam

import cv2
import numpy as np
import torch
import torch.nn.parallel

class FeatureExtractor():
    """ Class for extracting activations and registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs.append(x)
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targeted layers.
    3. Gradients from intermediate targeted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            elif "imp" in name.lower():
                continue
            else:
                x = module(x)
        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def get_attention_map(self, input, index=None, norm=None):
        if self.cuda:
            input = input.cuda()
        features, output = self.extractor(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=1)

        one_hot = np.zeros((output.size(0), output.size()[-1]), dtype=np.float32)
        for i in range(output.size(0)):
            one_hot[i][index[i]] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]

        target = features[-1]
        weights = torch.mean(grads_val, dim=(2, 3))

        cam = torch.zeros(target.shape[0], target.shape[2], target.shape[3])
        if self.cuda:
            cam = cam.cuda()

        for i in range(weights.size(1)):
            cam += weights[:, i].view(-1, 1, 1) * target[:, i, :, :]

        if norm == 'ReLU':
            cam = torch.relu(cam)
            cam = cam / (torch.max(cam) + 1e-6)
        elif norm == 'Sigmoid':
            cam = torch.sigmoid(cam)
        else:
            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-6)

        return cam, output

    def __call__(self, input, index=None, norm='ReLU'):
        if self.cuda:
            input = input.cuda()
        features, output = self.extractor(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=1)

        one_hot = np.zeros((output.size(0), output.size()[-1]), dtype=np.float32)
        for i in range(output.size(0)):
            one_hot[i][index[i]] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1].cpu().data.numpy()

        weights = np.mean(grads_val, axis=(2, 3))
        cam = np.zeros(target.shape[0:3], dtype=np.float32)

        for i in range(weights.shape[1]):
            cam += weights[:, i].reshape(-1, 1, 1) * target[:, i, :, :]

        if norm == 'ReLU':
            zeros = np.zeros(cam.shape, dtype=cam.dtype)
            cam = np.maximum(cam, zeros)
            cam = np.array([cv2.resize(c, (input.shape[2], input.shape[3])) for c in cam])
            cam = cam / (np.max(cam, axis=(1, 2), keepdims=True) + 1e-6)
        else:
            cam = cam - np.min(cam, axis=(1, 2), keepdims=True)
            cam = np.array([cv2.resize(c, (input.shape[2], input.shape[3])) for c in cam])
            cam = cam / (np.max(cam, axis=(1, 2), keepdims=True) + 1e-6)

        return cam
