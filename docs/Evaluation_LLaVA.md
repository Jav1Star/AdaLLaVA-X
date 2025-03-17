# Evaluation

In AdaLLaVA, we evaluate models on a existing benchmarks using official toolkit [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) to ensure the reproducibility.


## Evaluate on Custom Datasets

You can evaluate AdaLLaVA on your custom datasets by converting your dataset to original LLaVA's jsonl format, and evaluate using [`model_vqa_loader.py`](../src/adallava/eval/model_vqa_loader.py).


## Scripts
Below we provide an example for using original LLaVA evaluation script to evaluate following LLaVA evaluaton guidelines.

Before preparing task-specific data, **you MUST first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**. It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to `./playground/data/eval`. This also provides a general structure for all datasets.

### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `./playground/data/eval/vqav2`.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `./playground/data/eval/vqav2/answers_upload`.

