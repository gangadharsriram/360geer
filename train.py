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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, ray_elongation_loss
from utils.vgg_loss import vgg_loss
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, build_scaling_rotation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, match_mask_to_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import cv2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, fov_mod, sample_step, mask_path, sibr_mask_refcam=None,
             render_model='BEAP', raymap_path=None, focal_scaling=1.0, distortion_scaling=1.0, mirror_shift=0.0):
    if opt.densify_mode == "mcmc" and opt.cap_max <= 0:
        raise ValueError("--densify_mode mcmc requires --cap_max to be set (target max Gaussian count).")
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    render_dir = os.path.join(dataset.model_path, 'render')
    os.makedirs(render_dir, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree)

    dataset.fov_mod = fov_mod
    dataset.sample_step = sample_step
    dataset.render_model = render_model
    dataset.focal_scaling = focal_scaling
    dataset.distortion_scaling = distortion_scaling
    dataset.mirror_shift = mirror_shift
    dataset.raymap = None
    if raymap_path is not None and os.path.exists(raymap_path):
        dataset.raymap = np.load(raymap_path)
    scene = Scene(dataset, gaussians, shuffle=False)

    # Fixed ray-origin centroid for lambda_ray_aniso (utils/loss_utils.py's
    # ray_elongation_loss) - a single point stands in for "the camera" since
    # this investigation's narrow-baseline scenes have near-co-located
    # cameras (see the camera-baseline survey); computed once, not per-view.
    ray_aniso_cam_center = torch.stack([cam.camera_center for cam in scene.getTrainCameras()]).mean(dim=0)

    # Precomputed illumination-invariant structure priors (Structure Prior
    # Extraction/, CIConv 'W') for lambda_smooth's edge-aware weighting below -
    # loaded once into GPU memory, not per-iteration disk I/O. Resized to each
    # camera's actual training resolution since the priors are generated at
    # the raw dataset image resolution.
    smooth_prior_maps = {}
    if opt.smooth_prior_dir:
        for cam in scene.getTrainCameras():
            prior_path = os.path.join(opt.smooth_prior_dir, cam.image_name + ".png")
            prior = cv2.imread(prior_path, cv2.IMREAD_GRAYSCALE)
            if prior is None:
                raise FileNotFoundError(f"smooth_prior_dir set but no prior found at {prior_path}")
            prior = cv2.resize(prior, (cam.image_width, cam.image_height), interpolation=cv2.INTER_AREA)
            smooth_prior_maps[cam.image_name] = torch.from_numpy(prior).float().cuda() / 255.0

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Pre-compute viewer extra params so MiniCam uses the correct render mode.
    _render_model_map = {"BEAP": 0, "KB": 1, "EQ": 1, "PH": 2, "EQUI": 3}
    render_model_int = _render_model_map.get(render_model, 0)
    cam_extra_params: dict = {}
    if render_model in ("KB", "EQ", "PH"):
        train_cams = scene.getTrainCameras()
        if train_cams:
            ref_cam = train_cams[0]
            cam_extra_params["focal_x"] = ref_cam.focal_x
            cam_extra_params["focal_y"] = ref_cam.focal_y
            cam_extra_params["principal_x"] = ref_cam.principal_x
            cam_extra_params["principal_y"] = ref_cam.principal_y
            if render_model in ("KB", "EQ"):
                cam_extra_params["distortion_coeffs"] = ref_cam.distortion_coeffs
                cam_extra_params["raymap"] = ref_cam.raymap

    print("mask_path:", mask_path)
    valid_mask = None
    if mask_path is not None:
        valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        valid_mask = np.repeat(valid_mask[None, ...], 3, axis=0)
        valid_mask = torch.tensor(valid_mask)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    real3d_calibration_printed = False
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                extra_params = {"sample_step": sample_step, "render_model_int": render_model_int, **cam_extra_params}
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, width, height = network_gui.receive(extra_params)
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    if sibr_mask_refcam is not None:
                        print("Applying SIBR mask to network image {}".format(sibr_mask_refcam))
                        net_mask = custom_cam.get_viewpoint_mask(sibr_mask_refcam)
                        net_mask = torch.tensor(np.repeat(net_mask[None, ...], 3, axis=0))
                        net_image[net_mask == 0] = 0.0
                    net_image = torch.nn.functional.interpolate(net_image[None, ...], (height, width), mode='bilinear')[0]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if valid_mask is not None:
            image[match_mask_to_image(valid_mask, image) == 0] = 0.0
        # Loss
        gt_image = viewpoint_cam.sampled_image.cuda()
        if opt.row_weight_bottom_frac > 0 and opt.row_weight_value != 1.0:
            # Fixed per-row prior: down-weight the bottom rows (equirect nadir = capture rig / person, view-dependent content)
            Himg = image.shape[-2]; rw = torch.ones(Himg, device=image.device); rw[int(round(Himg * (1 - opt.row_weight_bottom_frac))):] = opt.row_weight_value
            w3 = rw[None, :, None].expand_as(image)
            Ll1 = (w3 * torch.abs(image - gt_image)).sum() / w3.sum()
            ssim_value = ssim(image, gt_image, mask=w3)
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        if opt.lambda_vgg > 0 and iteration >= opt.vgg_start_iter:
            loss = loss + opt.lambda_vgg * vgg_loss(image, gt_image, opt.vgg_crop)

        # Anisotropy regularization: penalize needle-thin Gaussians (large scale
        # ratio between the biggest and smallest axis). Hinge loss - ordinary thin
        # "pancake" Gaussians (flat surfaces) stay free below aniso_soft_ratio.
        scales = gaussians.get_scaling
        scale_ratio = scales.max(dim=1).values / scales.min(dim=1).values.clamp_min(1e-8)
        aniso_loss = torch.clamp(scale_ratio - opt.aniso_soft_ratio, min=0.0).mean()
        loss = loss + opt.lambda_aniso * aniso_loss

        # Ray-aligned elongation regularization: off by default (lambda=0, exact
        # no-op). Unlike lambda_aniso above (orientation-agnostic), this only
        # penalizes elongation along each Gaussian's own camera ray - the
        # specific axis narrow-baseline capture leaves unconstrained. See
        # utils/loss_utils.py's ray_elongation_loss for the full rationale.
        if opt.lambda_ray_aniso > 0:
            ray_loss = ray_elongation_loss(gaussians.get_xyz, gaussians.get_scaling,
                                            gaussians.get_rotation, ray_aniso_cam_center,
                                            max_ratio=opt.ray_aniso_max_ratio)
            loss = loss + opt.lambda_ray_aniso * ray_loss

        # Opacity-mass sparsity regularization: off by default (lambda=0, exact no-op).
        # One of the isolated single-variable experiments - see scripts/expX_*.sh.
        if opt.lambda_opacity_sparsity > 0:
            loss = loss + opt.lambda_opacity_sparsity * gaussians.get_opacity.mean()

        # Scale prior: off by default (lambda=0, exact no-op). L1 on the Gaussians' world-space scales (as in the MCMC
        # branch below, but available in the normal grad-threshold mode) - pushes back on oversized/floater Gaussians.
        if opt.lambda_scale_reg > 0:
            loss = loss + opt.lambda_scale_reg * gaussians.get_scaling.abs().mean()

        # Edge-aware image-space smoothness: off by default (lambda=0, exact no-op).
        # Penalizes local pixel-to-pixel variation in the RENDERED image, weighted
        # down wherever the ground truth itself has a real edge - so it discourages
        # the scattered/dotted redundant-Gaussian artifact (which shows up as
        # high-frequency noise in regions that should render flat) without
        # penalizing genuine detail the GT actually has there. Standard MonoDepth2-
        # style edge-aware smoothness formula, adapted from depth to RGB.
        if opt.lambda_smooth > 0:
            render_dx = torch.abs(image[:, :, :-1] - image[:, :, 1:])
            render_dy = torch.abs(image[:, :-1, :] - image[:, 1:, :])
            if opt.smooth_prior_dir:
                # Prior is already an edge-strength map (CIConv 'W', illumination-
                # invariant), not a raw image - use its value directly rather than
                # re-differentiating it. Averaged across each adjacent pixel pair
                # to align with render_dx/render_dy's shape.
                prior = smooth_prior_maps[viewpoint_cam.image_name]
                gt_dx = (0.5 * (prior[:, :-1] + prior[:, 1:]))[None]
                gt_dy = (0.5 * (prior[:-1, :] + prior[1:, :]))[None]
            else:
                gt_dx = torch.mean(torch.abs(gt_image[:, :, :-1] - gt_image[:, :, 1:]), dim=0, keepdim=True)
                gt_dy = torch.mean(torch.abs(gt_image[:, :-1, :] - gt_image[:, 1:, :]), dim=0, keepdim=True)
            smooth_loss = (render_dx * torch.exp(-gt_dx * opt.smooth_edge_scale)).mean() + \
                          (render_dy * torch.exp(-gt_dy * opt.smooth_edge_scale)).mean()
            loss = loss + opt.lambda_smooth * smooth_loss

        # MCMC regularizers (3D Gaussian Splatting as Markov Chain Monte Carlo) -
        # only active in --densify_mode mcmc. Separate knobs from
        # lambda_opacity_sparsity above (that one stays on raw get_opacity,
        # tuned for grad_threshold mode) - these target get_scaled_opacity
        # (what's actually rendered), matching how relocate_gs/add_new_gs
        # read opacity in scene/gaussian_model.py.
        if opt.densify_mode == "mcmc":
            loss = loss + opt.mcmc_opacity_reg * torch.abs(gaussians.get_scaled_opacity).mean()
            loss = loss + opt.mcmc_scale_reg * torch.abs(gaussians.get_scaling).mean()


        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset.train_test_exp, valid_mask, render_dir)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if opt.densify_mode == "mcmc":
                if iteration > opt.densify_from_iter and iteration < opt.densify_until_iter \
                        and iteration % opt.densification_interval == 0:
                    dead_mask = (gaussians.get_scaled_opacity <= opt.min_opacity).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask)
                    gaussians.add_new_gs(cap_max=opt.cap_max)
            elif iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if opt.densify_stats_real3d_distscaled:
                    gaussians.add_densification_stats_real3d_distscaled(visibility_filter, viewpoint_cam.camera_center)
                elif opt.densify_stats_real3d:
                    gaussians.add_densification_stats_real3d(visibility_filter)
                elif opt.densify_stats_2d:
                    gaussians.add_densification_stats_direct(viewspace_point_tensor.grad, visibility_filter)
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if (opt.densify_stats_real3d or opt.densify_stats_real3d_distscaled) and not real3d_calibration_printed:
                        # One-time calibration print: the real gradient's scale is
                        # unknown ahead of time and unrelated to the proxy statistic's
                        # tuned threshold values - report it once so the right
                        # densify_grad_threshold can be picked with evidence, not a guess.
                        label = "densify_stats_real3d_distscaled" if opt.densify_stats_real3d_distscaled else "densify_stats_real3d"
                        stat = gaussians.xyz_gradient_accum / gaussians.denom.clamp(min=1)
                        print(f"[{label} calibration] iter={iteration} mean={stat.mean().item():.8f} "
                              f"median={stat.median().item():.8f} p90={stat.quantile(0.9).item():.8f} "
                              f"max={stat.max().item():.8f}")
                        real3d_calibration_printed = True
                    size_threshold = 20 if iteration > opt.size_prune_from_iter else None
                    aniso_threshold = opt.aniso_hard_ratio if iteration > opt.aniso_prune_from_iter else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold, aniso_threshold, bool(opt.prune_by_extent))

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            elif iteration < opt.prune_until_iter:
                # Cloning/splitting has stopped (past densify_until_iter), but keep
                # cleaning up redundant/oversized survivors instead of freezing the
                # population for the rest of training.
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.size_prune_from_iter else None
                    aniso_threshold = opt.aniso_hard_ratio if iteration > opt.aniso_prune_from_iter else None
                    gaussians.prune_only(opt.min_opacity, scene.cameras_extent, size_threshold, aniso_threshold, bool(opt.prune_by_extent))

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if opt.densify_mode == "mcmc":
                    # 3DGS-MCMC's SGLD position-noise exploration term - runs every
                    # iteration (not gated by densification_interval), scaled by each
                    # Gaussian's own covariance and gated by opacity so well-formed
                    # (high rendered-opacity) Gaussians get little/no noise while
                    # low-opacity ones get a full random-walk kick. get_scaled_opacity
                    # (rendered), not get_opacity, gates this - see the comment block
                    # above relocate_gs in gaussian_model.py for why.
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))

                    noise = torch.randn_like(gaussians._xyz) * op_sigmoid(1 - gaussians.get_scaled_opacity) * opt.mcmc_noise_lr * xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, valid_mask, render_dir=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    if valid_mask is not None:
                        image[match_mask_to_image(valid_mask, image) == 0] = 0.0
                    gt_image = torch.clamp(viewpoint.sampled_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # Fixed-subset progress dump (every 5th test camera, UT-ODGS
                    # convention) - replaces the old training_snapshots/ dir (random
                    # training view every 500 iters, no GT, nothing fixed to compare
                    # across checkpoints).
                    if render_dir is not None and config['name'] == 'test' and idx % 5 == 0:
                        def to_bgr_u8(t):
                            return (t.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
                        gt_path = os.path.join(render_dir, f"gt_test_{viewpoint.image_name}.png")
                        if not os.path.exists(gt_path):
                            cv2.imwrite(gt_path, to_bgr_u8(gt_image))
                        cv2.imwrite(os.path.join(render_dir, f"iteration_test_{iteration}_{viewpoint.image_name}.png"), to_bgr_u8(image))
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image[None], gt_image[None]).mean().double()
                    lpips_test += lpips(image[None], gt_image[None], net_type='vgg').mean().double()  # same call as metrics.py
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test), flush=True)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 7000, 15000, 20000, 25000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--mask_path", type=str, default = None)
    parser.add_argument("--sibr_mask_refcam", type=str, default = None)
    parser.add_argument("--sample_step", type=float, default=0.002)
    parser.add_argument("--fov_mod", type=float, default = None)
    parser.add_argument("--render_model", type=str, default='BEAP',
                        help="Rendering/training projection mode: BEAP (default), KB, EQ, PH, or EQUI (equirectangular)")
    parser.add_argument("--raymap_path", type=str, default=None,
                        help="Path to a .npy per-pixel ray-direction map (required for KB/EQ training)")
    parser.add_argument("--focal_scaling", type=float, default=1.0,
                        help="Scale factor applied to the focal length (KB/PH modes)")
    parser.add_argument("--distortion_scaling", type=float, default=1.0,
                        help="Scale factor applied to distortion coefficients (KB mode; 0 = EQ)")
    parser.add_argument("--mirror_shift", type=float, default=0.0,
                        help="Mirror-model shift parameter xi for omnidirectional mapping (KB mode)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, \
             args.fov_mod,  args.sample_step, args.mask_path, args.sibr_mask_refcam, \
             args.render_model, args.raymap_path, args.focal_scaling, args.distortion_scaling, args.mirror_shift)

    # All done
    print("\nTraining complete.")
