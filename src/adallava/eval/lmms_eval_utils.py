import inspect
from datasets import Dataset
from typing import List, Optional, Tuple, Union
from lmms_eval.api.metrics import stderr_for_metric
from lmms_eval.evaluator_utils import TaskOutput
from lmms_eval.api.instance import Instance
from lmms_eval.api.filter import Filter, FilterEnsemble

EFFICIENCY_METRICS = ["flops", 
                      "avg_flops", 
                      "prefill_flops", 
                      "prefill_time",
                      "memory_consumption", 
                      "prefill_memory_consumption"]


class GetEfficiencyRecord:
    def __init__(self, task_output):
        self.task_output = task_output

    def apply(self, instances: List[Instance], docs: List[Dataset]) -> None:
        resps = [inst.resps for inst in instances]  # operate just on the model responses
        
        for inst, resp in zip(instances, resps):
            inst.resps = resp[0]['resp']
            for metric in EFFICIENCY_METRICS:
                self.task_output.sample_metrics[(metric, metric)].append(resp[0][metric])


def efficiency_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A value
    """
    try:
        avg_value = sum(results) / len(results)
    except:
        avg_value = 0

    return avg_value


class TaskOutputWithEfficiency(TaskOutput):
    
    @classmethod
    def from_taskdict(cls, task_name: str, task):
        task_output = super().from_taskdict(task_name, task)

        if hasattr(task_output.task, "_filters"):
            task_output.task._filters = [GetEfficiencyRecord(task_output)] + task_output.task._filters
        else:
            task_output.task._filters = [GetEfficiencyRecord(task_output)]

        efficiency_aggregate_list = {metric: efficiency_aggregate_results for metric in EFFICIENCY_METRICS}
        
        if hasattr(task_output.task, '_aggregation_list'):
            task_output.task._aggregation_list.update(efficiency_aggregate_list)
        else:
            task_output.task._aggregation_list = efficiency_aggregate_list
            
        return task_output


def get_task_list(task_dict: dict) -> List[TaskOutputWithEfficiency]:
    outputs = []
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            _outputs = get_task_list(task_obj)
            outputs.extend(_outputs)
        else:
            task_output = TaskOutputWithEfficiency.from_taskdict(task_name, task_obj)
            outputs.append(task_output)

    return outputs
