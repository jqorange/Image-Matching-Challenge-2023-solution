import torch
import kornia.feature as KF

class KeyNetAffNetHardNet(KF.LocalFeature):
    """Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.

    .. image:: _static/img/keynet_affnet.jpg
    """

    def __init__(
            self,
            num_features: int = 5000,
            upright: bool = False,
            device=torch.device('cpu'),
            scale_laf: float = 1.0,
    ):
        ori_module = KF.PassLAF() if upright else KF.LAFOrienter(angle_detector=KF.OriNet(False)).eval()
        if not upright:
            weights = torch.load('../params/OriNet.pth')['state_dict']
            ori_module.angle_detector.load_state_dict(weights)
        detector = KF.KeyNetDetector(
            False, num_features=num_features, ori_module=ori_module, aff_module=KF.LAFAffNetShapeEstimator(False).eval()
        ).to(device)
        kn_weights = torch.load('../params/keynet_pytorch.pth')['state_dict']
        detector.model.load_state_dict(kn_weights)
        affnet_weights = torch.load('../params/AffNet.pth')['state_dict']
        detector.aff.load_state_dict(affnet_weights)

        hardnet = KF.HardNet(False).eval()
        hn_weights = torch.load('../params/HardNetLib.pth')['state_dict']
        hardnet.load_state_dict(hn_weights)
        descriptor = KF.LAFDescriptor(hardnet, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor, scale_laf)