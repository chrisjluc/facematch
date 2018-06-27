from .. import consts

from ..task_manager import TaskManager
from ..tasks import GPUTask

import pytest


class TestTask(GPUTask):
    def __init__(self):
        pass

    def run(self):
        for i in range(50000000):
            pass


class TestTaskManager(object):

    def test_single_task(self):
        tasks = [TestTask()]
        manager = TaskManager(tasks)
        manager.run_tasks()

    def test_four_tasks(self):
        tasks = [TestTask(), TestTask(), TestTask(), TestTask()]
        manager = TaskManager(tasks)
        manager.run_tasks()

    def test_six_tasks(self):
        tasks = [TestTask(), TestTask(), TestTask(), TestTask(), TestTask(), TestTask()]
        manager = TaskManager(tasks)
        manager.run_tasks()

