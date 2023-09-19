# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import HYBRIDS, build_posenet, build_detector, build_classifier, build_optical, build_simulator
import torch.nn as nn
from mmcv.runner import BaseModule
from collections import OrderedDict
import torch
import torch.distributed as dist
from mmdet.core import encode_mask_results
from torchvision import transforms
import torch.nn.functional as F


@HYBRIDS.register_module()
class BaseHybrid_sim(BaseModule):   # with simulator
    def __init__(self,
                 optical,
                 simulator,
                 classifier=None,
                 detector=None,
                 posenet=None,
                 init_cfg=None
                 ):
        super().__init__(init_cfg=init_cfg)

        self.optical = build_optical(optical)
        self.simulator = build_simulator(simulator)
        if classifier is not None:
            self.classifier = build_classifier(classifier)
        if detector is not None:
            self.detector = build_detector(detector)
        if posenet is not None:
            self.posenet = build_posenet(posenet)

    def forward_train(self, input_dict):
        """Forward computation during training.
        """
        losses_all = {}
        if hasattr(self, 'classifier'):
            data = input_dict['cls']
            data['img'] = self.optical(data['img'])
            #print(data['img'].size())
            img = self.simulator(data['img'])
            #resize = transforms.Resize([112,96])
            #img = resize(img)
            img = F.interpolate(img, size=(112, 96)) #resize
            data['img'] = img.repeat(1, 3, 1, 1)   # repeat 1 to 3 channel
            #print(data['img'].size())
            losses = self.classifier(**data, return_loss=True)
            for key in losses.keys():
                losses_all['cls_'+key] = losses[key]

        if hasattr(self, 'detector'):
            data = input_dict['det']
            data['img'] = F.interpolate(data['img'], size=(507,507))  #resizee
            data['img'] = self.optical(data['img'])
            #data['img'] = self.simulator(data['img'])
            #print(data['img'].size())
            img = self.simulator(data['img'])
            data['img'] = img.repeat(1, 3, 1, 1) # repeat channel
            #print(data['img'].size())
            losses = self.detector(**data, return_loss=True)
            for key in losses.keys():
                losses_all['det_'+key] = losses[key]

        if hasattr(self, 'posenet'):
            data = input_dict['pose']
            data['img'] = self.optical(data['img'])
            #data['img'] = self.simulator(data['img'])
            img = self.simulator(data['img'])
            data['img'] = img.repeat(1, 3, 1, 1)  # repeat channel
            losses = self.posenet(**data, return_loss=True)
            for key in losses.keys():
                losses_all['pose_'+key] = losses[key]
        return losses_all

    def forward_test(self, input_dict):
        num_batch = 0
        for key in input_dict.keys():
            num_batch = max(num_batch, len(input_dict[key]['img']))

        results = [{}] * num_batch
        if 'cls' in input_dict.keys():
            data = input_dict['cls']
            img = data['img']
            if img.dim() == 5:
                b, n, _, _, _ = img.shape
                img = img.flatten(0, 1)
                img = self.optical(img)
                img = self.simulator(img)
                img = F.interpolate(img, size=(112, 96))   # resize
                img = img.repeat(1, 3, 1, 1)   # repeat channel
                _, c, h, w = img.shape
                img = img.reshape(b, n, c, h, w)
                data['img'] = img
            else:
                img = self.optical(img)
                #data['img'] = self.simulator(img)
                img = self.simulator(img)
                img = F.interpolate(img, size=(112, 96))   # resize
                data['img'] = img.repeat(1, 3, 1, 1)   # repeat channel
            result = self.classifier(**data, return_loss=False)
            for i in range(len(result)):
                results[i]['cls_result'] = result[i]

        if 'det' in input_dict.keys():
            data = input_dict['det']
            img = data['img']
            if isinstance(img, list):
                img = [self.optical(tmp) for tmp in img]
                img = [self.simulator(F.interpolate(tmp, size=(507, 507))) for tmp in img]   ##
                img = [tmp.repeat(1, 3, 1, 1) for tmp in img]   # repeat channel
                data['img'] = img
            else:
                img = self.optical(img)
                #data['img'] = self.simulator(img)
                img = self.simulator(F.interpolate(img, size=(507, 507)))   ##
                data['img'] = img.repeat(1, 3, 1, 1)      ##
            result = self.detector(**data, return_loss=False, rescale=True)
            #print(result)
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

            for i in range(len(result)):
                results[i]['det_result'] = result[i]

        if 'pose' in input_dict.keys():
            data = input_dict['pose']
            img = self.optical(data['img'])
            img = self.simulator(img)
            data['img'] = img.repeat(1, 3, 1, 1)  ##
            result = self.posenet(**data, return_loss=False)
            results[0]['pose_result'] = result

        return results

    def forward(self, input_dict, return_loss=True):
        if return_loss:
            return self.forward_train(input_dict)
        else:
            return self.forward_test(input_dict)

    def init_weights(self):
        self.optical.init_weights()
        #self.simulator.init_weights()
        if hasattr(self, 'classifier'):
            self.classifier.init_weights()

        if hasattr(self, 'detector'):
            self.detector.init_weights()

        if hasattr(self, 'posenet'):
            self.posenet.init_weights()

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars \
                contains all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, float):
                log_vars[loss_name] = loss_value
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors or float')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if not isinstance(loss_value, float):
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()
            else:
                log_vars[loss_name] = loss_value

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        input_dict = data['input_dict']
        outputs = dict(
            loss=loss, log_vars=log_vars,
            num_samples=len(input_dict[list(input_dict.keys())[0]]['img_metas']))
        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        input_dict = data['input_dict']

        outputs = dict(
            loss=loss, log_vars=log_vars,
            num_samples=len(input_dict[list(input_dict.keys())[0]]['img_metas']))
        return outputs
