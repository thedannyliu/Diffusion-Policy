"""
Simple synchronous vector environment for Push-T evaluation.
This is a minimal implementation that works with both gym and gymnasium.
"""

import numpy as np
from typing import List, Callable, Optional, Tuple, Any


class SimpleSyncVectorEnv:
    """
    Simple synchronous vectorized environment.
    Runs environments sequentially in the main process.
    Compatible with both gym 0.22 and gymnasium APIs.
    """
    
    def __init__(self, env_fns: List[Callable]):
        """
        Args:
            env_fns: List of callables that create environments
        """
        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        
        # Get spaces from first environment
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.metadata = self.envs[0].metadata
        
    def reset(self, seed=None, **kwargs):
        """Reset all environments. Returns obs only (gym 0.21 style)."""
        observations = []
        
        for i, env in enumerate(self.envs):
            if seed is not None:
                if isinstance(seed, (list, tuple)):
                    env_seed = seed[i]
                else:
                    env_seed = seed + i
                try:
                    obs = env.reset(seed=env_seed)
                except TypeError:
                    # Old gym API doesn't take seed in reset
                    env.seed(env_seed)
                    obs = env.reset()
            else:
                obs = env.reset()
            
            # Handle both (obs,) and (obs, info) return formats
            if isinstance(obs, tuple):
                observations.append(obs[0])
            else:
                observations.append(obs)
        
        # Stack observations (handles dict obs)
        return self._stack_obs(observations)
    
    def step(self, actions):
        """Step all environments. Returns (obs, rewards, dones, infos) - gym 0.21 style."""
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            result = env.step(action)
            
            # Handle both old (obs, rew, done, info) and new (obs, rew, term, trunc, info) APIs
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, done, truncated, info = result
                # Combine terminated and truncated
                done = done or truncated
            
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (
            self._stack_obs(observations),
            np.array(rewards),
            np.array(dones),
            infos
        )
    
    def _stack_obs(self, observations):
        """Stack observations into batch format."""
        if isinstance(observations[0], dict):
            # Dict observations
            stacked = {}
            for key in observations[0].keys():
                stacked[key] = np.stack([obs[key] for obs in observations])
            return stacked
        else:
            return np.stack(observations)
    
    def call(self, method_name: str, *args, **kwargs):
        """Call a method on all environments."""
        results = []
        for env in self.envs:
            method = getattr(env, method_name)
            results.append(method(*args, **kwargs))
        return results
    
    def call_each(self, method_name: str, 
                  args_list: list = None, 
                  kwargs_list: list = None):
        """Call a method on each environment with different arguments."""
        n_envs = self.num_envs
        if args_list is None:
            args_list = [tuple()] * n_envs
        assert len(args_list) == n_envs
        
        if kwargs_list is None:
            kwargs_list = [dict()] * n_envs
        assert len(kwargs_list) == n_envs
        
        results = []
        for i, env in enumerate(self.envs):
            method = getattr(env, method_name)
            result = method(*args_list[i], **kwargs_list[i])
            results.append(result)
        return results
    
    def call_async(self, method_name: str, *args, **kwargs):
        """Async call (just stores for later)."""
        self._pending_call = (method_name, args, kwargs)
    
    def call_wait(self):
        """Wait for async call."""
        method_name, args, kwargs = self._pending_call
        return self.call(method_name, *args, **kwargs)
    
    def set_attr(self, name: str, values):
        """Set attribute on all environments."""
        if not isinstance(values, (list, tuple)):
            values = [values] * self.num_envs
        for env, value in zip(self.envs, values):
            setattr(env, name, value)
    
    def render(self, mode='rgb_array'):
        """Render all environments - return video paths from VideoRecordingWrapper."""
        results = []
        for env in self.envs:
            # Try to call render without arguments first (gym 0.22+ style)
            # which returns the video file path for VideoRecordingWrapper
            try:
                result = env.render()
            except TypeError:
                # Fall back to old style with mode argument
                result = env.render(mode=mode)
            results.append(result)
        return results
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            try:
                env.close()
            except:
                pass
    
    def __len__(self):
        return self.num_envs
