#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = "./output/"
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._camera_model = "FISHEYE" #FISHEYE/PINHOLE
        self.dataset = "AUTO" #AUTO/COLMAP/BLENDER/SCANNETPP/EQUIRECT
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cpu" #"cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        # Both off by default (preserve current behavior). Independently toggle the two
        # pieces of the densification gradient-norm statistic's inflation relative to
        # OmniGS/vanilla 3DGS - split apart after exp59/60 (both on at once) overshot
        # badly, to find which one is actually responsible before assuming both need
        # to change together.
        self.densify_grad_signed = False    # False: fabsf() every per-pixel contribution
                                             # before summing (never cancels). True: signed
                                             # accumulation, matching OmniGS (opposing pushes
                                             # cancel).
        self.densify_grad_unscaled = False  # False: scale by (depth^2+1)*0.3 before
                                             # comparing against densify_grad_threshold,
                                             # inflating far/background content's statistic
                                             # more than near content's. True: no scaling,
                                             # matching OmniGS/vanilla 3DGS (no equivalent
                                             # term at all).
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01 
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 300
        self.opacity_reset_interval = 3000
        self.size_prune_from_iter = 3000  # was implicitly tied to opacity_reset_interval;
                                           # split out so the two can be tuned independently
        self.prune_by_extent = 1          # world-space size-cull on/off, independent of the
                                           # screen-space one (max_radii2D) above - mirrors
                                           # OmniGS's prune_by_extent flag. int not bool: this
                                           # ParamGroup maps bool defaults to store_true CLI
                                           # flags, which can't be turned off from a True
                                           # default; --prune_by_extent 0 disables it.
        self.aniso_prune_from_iter = 3000 # own timing gate for the aniso hard-cutoff, split
                                           # out from size_prune_from_iter - aniso is a
                                           # 360geer-specific criterion OmniGS doesn't have,
                                           # so it must not share size_prune_from_iter's gate
                                           # (defaults match, so behavior is unchanged unless
                                           # one is overridden independently)
        self.densify_stats_2d = 0         # off by default (preserves current behavior): the
                                           # densification gradient-norm statistic currently
                                           # reads 3 components of viewspace_point_tensor.grad
                                           # (gaussian_model.py's add_densification_stats), one
                                           # more than OmniGS/vanilla 3DGS's 2 - inflating the
                                           # value compared against densify_grad_threshold with
                                           # no corresponding threshold adjustment. Set to 1 to
                                           # use the already-present add_densification_stats_direct
                                           # (2-component) instead. Does NOT fix the deeper
                                           # cause (submodules/geer-rasterizer's CUDA backward
                                           # applies fabsf() before accumulating, so pushes never
                                           # cancel the way OmniGS's signed accumulation does) -
                                           # this only removes the 3rd-component contribution.
        self.densify_stats_real3d = 0      # off by default (preserves current behavior).
                                            # Set to 1 to trigger clone/split off
                                            # self._xyz.grad (the real world-space position
                                            # gradient, matching the 3DGEER paper's stated
                                            # "gradients of the view-space mean3D" design -
                                            # p.27 "Implementation Details") instead of the
                                            # fabsf/depth-scaled dL_dmean2D proxy. Note the
                                            # value scale is completely different from the
                                            # proxy's - densify_grad_threshold needs fresh
                                            # calibration when this is on, the existing
                                            # tuned values (e.g. 0.00006) do not carry over.
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.prune_until_iter = 15000     # was implicitly == densify_until_iter (pruning
                                           # stopped dead the moment cloning/splitting did);
                                           # split out so cleanup can keep running afterward
        self.densify_grad_threshold = 0.0002
        self.min_opacity = 0.005       # densify_and_prune opacity floor (was hardcoded 0.005)
        self.lambda_aniso = 0.01       # weight of the soft scale-anisotropy regularizer
        self.aniso_soft_ratio = 10.0   # scale ratios (max/min axis) below this are free
        self.aniso_hard_ratio = 50.0   # densify_and_prune hard cutoff on scale ratio
        self.lambda_opacity_sparsity = 0.0  # off by default - see train.py loss section
        self.lambda_smooth = 0.0       # off by default - edge-aware image smoothness, see train.py
        self.smooth_edge_scale = 10.0  # GT-gradient sensitivity for the edge-aware weighting above
                                        # (standard MonoDepth2 default) - higher means edges need to
                                        # be sharper in GT before smoothness backs off there
        self.depth_l1_weight_init = 0.1
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
