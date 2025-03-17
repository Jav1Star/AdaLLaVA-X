import lmms_eval.__main__ as lmms_eval_main
from lmms_eval.api.registry import get_model
import src.adallava.eval.adallava_wrapper
from src.adallava.eval.lmms_eval_utils import get_task_list

lmms_eval_main.evaluator.get_model = get_model
lmms_eval_main.evaluator.get_task_list = get_task_list

if __name__ == "__main__":
    lmms_eval_main.cli_evaluate()