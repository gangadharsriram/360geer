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
                                            # Tested and rejected as a densification signal on
                                            # its own (exp94-97) - too weak/noisy in exactly the
                                            # far/ambiguous regions where densification matters
                                            # most. See densify_stats_real3d_distscaled below,
                                            # which compensates for that specific weakness.
        self.densify_stats_real3d_distscaled = 0  # off by default. Set to 1 to use
                                            # add_densification_stats_real3d_distscaled: same
                                            # real self._xyz.grad as densify_stats_real3d above,
                                            # but each iteration's per-Gaussian gradient norm is
                                            # multiplied by that Gaussian's distance to the
                                            # current camera before accumulating (matching
                                            # 3DGRT/gsplat's GSStrategy: accumulates
                                            # ||grad_position|| * distance_to_camera / 2). Real,
                                            # signed gradient (not the fabsf/depth-scaled proxy),
                                            # but compensates for the real gradient's weak signal
                                            # at distance the same way 3DGEER's depth^2 factor
                                            # does for the proxy - unlike plain densify_stats_real3d,
                                            # this doesn't fall down in far/ambiguous regions.
                                            # Own threshold calibration needed, same as
                                            # densify_stats_real3d - values do not carry over
                                            # from either the proxy's or plain real3d's tuning.
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
        self.lambda_ray_aniso = 0.0    # off by default. Weight of the ray-aligned elongation
                                        # penalty (utils/loss_utils.py's ray_elongation_loss) -
                                        # unlike lambda_aniso above (any max/min scale ratio,
                                        # regardless of orientation), this only fires on
                                        # elongation aligned with each Gaussian's own camera
                                        # ray - the specific axis narrow-baseline capture
                                        # leaves unconstrained, and the direction the needle-
                                        # shaped scattered-artifact Gaussians actually stretch
                                        # along. Does not flatten legitimate thin structures
                                        # lying sideways to the camera (wires, edges).
        self.ray_aniso_max_ratio = 3.0 # elongation along-ray vs perpendicular ratio above
                                        # which the penalty kicks in (hinge, like aniso_soft_ratio)
        self.lambda_opacity_sparsity = 0.0  # off by default - see train.py loss section
        self.row_weight_bottom_frac = 0.0  # off by default. Fraction of image rows at the bottom (equirect nadir) whose L1/SSIM loss weight is row_weight_value
        self.row_weight_value = 1.0        # weight applied to those rows (others 1.0); weighted mean, so the loss scale is unchanged
        self.lambda_vgg = 0.0          # off by default. Weight of the VGG16 perceptual loss (utils/vgg_loss.py)
        self.vgg_start_iter = 0        # VGG loss only applies from this iteration on (e.g. 15000 = after densification ends)
        self.vgg_crop = 512            # VGG loss is computed on one random crop x crop patch per step (0 = full image)
        self.lambda_scale_reg = 0.0    # off by default. Weight on mean(|Gaussian scale|) (world units) in the normal densify mode, see train.py
        self.lambda_smooth = 0.0       # off by default - edge-aware image smoothness, see train.py
        self.smooth_edge_scale = 10.0  # GT-gradient sensitivity for the edge-aware weighting above
                                        # (standard MonoDepth2 default) - higher means edges need to
                                        # be sharper in GT before smoothness backs off there
        self.smooth_prior_dir = ""     # off by default (""). If set, the edge-aware weighting
                                        # above uses precomputed illumination-invariant structure
                                        # maps (Structure Prior Extraction/, CIConv 'W') from this
                                        # dir instead of the raw GT gradient - see train.py.
        self.depth_l1_weight_init = 0.1
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.densify_mode = "grad_threshold"  # "grad_threshold" (default, current
                                               # behavior unchanged, everything above) or
                                               # "mcmc" (3DGS-MCMC-style relocation/growth,
                                               # see gaussian_model.py's relocate_gs/
                                               # add_new_gs - replaces the whole
                                               # densify_and_prune call in train.py, does
                                               # not stack with it).
        self.cap_max = -1              # required when densify_mode=mcmc: target max
                                        # Gaussian count grown toward via add_new_gs.
                                        # No effect in grad_threshold mode.
        self.mcmc_noise_lr = 5e5       # mcmc mode only: scale of the per-iteration,
                                        # covariance-shaped, opacity-gated position noise
                                        # (3DGS-MCMC's SGLD exploration term)
        self.mcmc_opacity_reg = 0.01   # mcmc mode only: weight on mean(|rendered opacity|)
                                        # - separate knob from lambda_opacity_sparsity
                                        # above (that one stays tuned to 0.012 on raw
                                        # get_opacity for grad_threshold mode; this one
                                        # targets get_scaled_opacity, only applied in
                                        # mcmc mode)
        self.mcmc_scale_reg = 0.01     # mcmc mode only: weight on mean(|scale|)
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
