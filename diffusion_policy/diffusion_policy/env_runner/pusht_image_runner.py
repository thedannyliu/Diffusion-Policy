import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import matplotlib.pyplot as plt
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.simple_vec_env import SimpleSyncVectorEnv
# from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class PushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = SimpleSyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_coverages = [None] * n_inits
        all_dones = [None] * n_inits
        all_step_counts = [None] * n_inits
        all_final_distances = [None] * n_inits
        all_block_trajectories = [None] * n_inits
        all_agent_trajectories = [None] * n_inits
        
        # Track inference latency for fair comparison
        all_latencies = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                    # Track latency if available
                    if hasattr(policy, '_last_latency_ms'):
                        all_latencies.append(policy._last_latency_ms)

                # device_transfer - only convert tensors, skip floats like latency_ms
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy() if torch.is_tensor(x) else x)

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            video_paths = env.render()
            rewards_list = env.call('get_attr', 'reward')
            dones_list = env.call('get_attr', 'done')
            infos_list = env.call('get_infos')

            # Collect per-episode data
            for local_idx in range(this_n_active_envs):
                global_idx = start + local_idx
                if global_idx >= n_inits:
                    break

                all_video_paths[global_idx] = video_paths[local_idx]
                all_rewards[global_idx] = rewards_list[local_idx]
                all_dones[global_idx] = dones_list[local_idx]

                info = infos_list[local_idx] if local_idx < len(infos_list) else dict()
                block_traj = np.array(info.get('block_pose', []))
                agent_traj = np.array(info.get('pos_agent', []))
                coverage_traj = np.array(info.get('coverage', []))

                all_coverages[global_idx] = coverage_traj
                all_block_trajectories[global_idx] = block_traj
                all_agent_trajectories[global_idx] = agent_traj

                # Step count for this rollout
                all_step_counts[global_idx] = len(rewards_list[local_idx])

                # Final distance between block center and goal center
                if block_traj.shape[0] > 0 and isinstance(info.get('goal_pose', None), (list, np.ndarray)):
                    goal_pose = np.array(info['goal_pose'])
                    # Use last block pose
                    block_final = block_traj[-1]
                    all_final_distances[global_idx] = np.linalg.norm(
                        block_final[:2] - goal_pose[:2]
                    )
                else:
                    all_final_distances[global_idx] = np.nan
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        max_coverages = collections.defaultdict(list)
        success_rates = collections.defaultdict(list)
        final_distances = collections.defaultdict(list)
        step_counts = collections.defaultdict(list)
        smoothness_values = collections.defaultdict(list)

        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            rewards = np.array(all_rewards[i]) if all_rewards[i] is not None else np.array([])
            coverages = np.array(all_coverages[i]) if all_coverages[i] is not None else np.array([])
            dones = np.array(all_dones[i]) if all_dones[i] is not None else np.array([])

            # Aggregate reward-based score (same as original DP code)
            max_reward = float(np.max(rewards)) if rewards.size > 0 else 0.0
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # Aggregate coverage (max over episode, based on cached coverage)
            if coverages.size > 0:
                max_cov = float(np.max(coverages))
                max_coverages[prefix].append(max_cov)

            # Success indicator: whether episode ever satisfies success condition
            if dones.size > 0:
                success = float(np.max(dones))
            else:
                # Fallback: treat saturated reward as success
                success = float(max_reward >= 1.0 - 1e-6)
            success_rates[prefix].append(success)

            # Final distance and step count
            if all_final_distances[i] is not None:
                final_distances[prefix].append(float(all_final_distances[i]))
            if all_step_counts[i] is not None:
                step_counts[prefix].append(int(all_step_counts[i]))

            # Simple smoothness metric based on block trajectory
            block_traj = all_block_trajectories[i]
            if block_traj is not None and isinstance(block_traj, np.ndarray) and block_traj.shape[0] >= 3:
                vel = np.diff(block_traj[:, :2], axis=0)
                acc = np.diff(vel, axis=0)
                jerk = np.sum(acc ** 2, axis=1)
                smoothness = float(np.mean(jerk))
                smoothness_values[prefix].append(smoothness)
            else:
                smoothness_values[prefix].append(0.0)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            log_data[name] = float(np.mean(value))

        # Log additional aggregate metrics
        for prefix in max_rewards.keys():
            if len(max_coverages[prefix]) > 0:
                log_data[prefix+'target_area_coverage'] = float(np.mean(max_coverages[prefix]))
            if len(success_rates[prefix]) > 0:
                log_data[prefix+'success_rate'] = float(np.mean(success_rates[prefix]))
            if len(final_distances[prefix]) > 0:
                log_data[prefix+'final_distance'] = float(np.mean(final_distances[prefix]))
            if len(step_counts[prefix]) > 0:
                log_data[prefix+'mean_step_count'] = float(np.mean(step_counts[prefix]))
            if len(smoothness_values[prefix]) > 0:
                log_data[prefix+'smoothness'] = float(np.mean(smoothness_values[prefix]))
        
        # log latency statistics (for fair comparison between FM and DDPM)
        if len(all_latencies) > 0:
            latencies = np.array(all_latencies)
            log_data['inference_latency_ms'] = np.mean(latencies)
            log_data['inference_latency_p50_ms'] = np.percentile(latencies, 50)
            log_data['inference_latency_p95_ms'] = np.percentile(latencies, 95)

        # Generate a few trajectory plots for qualitative analysis (test rollouts)
        try:
            n_traj_plots = 3
            plot_count = 0
            for idx in range(n_inits):
                if self.env_prefixs[idx] != 'test/':
                    continue
                if plot_count >= n_traj_plots:
                    break
                block_traj = all_block_trajectories[idx]
                agent_traj = all_agent_trajectories[idx]
                if block_traj is None or agent_traj is None:
                    continue
                seed = self.env_seeds[idx]

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.plot(block_traj[:, 0], block_traj[:, 1], '-o', label='block')
                ax.plot(agent_traj[:, 0], agent_traj[:, 1], '-x', label='agent')
                ax.set_title(f'Test trajectory (seed={seed})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_aspect('equal', 'box')
                ax.legend()
                fig.tight_layout()

                fig_path = pathlib.Path(self.output_dir).joinpath(f"trajectory_seed_{seed}.png")
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path)
                plt.close(fig)

                log_data[f'test/trajectory_plot_{seed}'] = wandb.Image(str(fig_path))
                plot_count += 1
        except Exception:
            # Do not fail evaluation if plotting fails (e.g., no DISPLAY)
            pass

        return log_data
