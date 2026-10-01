import unittest
from diffusion_policy.common.task_success import episode_task_success


class TaskSuccessTest(unittest.TestCase):
    def test_time_limit_termination_is_not_success(self):
        self.assertFalse(episode_task_success({"success": [False, False], "done": [False, True]}))

    def test_any_task_success_counts(self):
        self.assertTrue(episode_task_success({"success": [False, True, False]}))

    def test_missing_or_empty_history_is_not_silently_scored(self):
        for info in ({}, {"success": []}):
            with self.subTest(info=info), self.assertRaises(ValueError):
                episode_task_success(info)
