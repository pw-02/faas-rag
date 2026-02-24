import re
import numpy as np
import warnings
from collections import Counter
from pwrag.evaluator.utils import normalize_answer
from pwrag.args.args import AppConfig
class BaseMetric:
    """`BaseMetric` serves as the base object of all metrics. Implemented metric should
    inherit this class.
    """

    metric_name = "base"

    def __init__(self, config: AppConfig):
        self.config = config
        self.dataset_name = config.dataset.dataset_name

    def calculate_metric(self, data):
        """Get the total score of this metric and score for each sample.

        Args:
            data object: it contains basic information and generated information.

        Returns:
            (metric_score: dict, metric_score_list: list)
            metric_score: such as ``{'em': 0.53}``.
            metric_score_list: score for each sample.

        """
        return {}, []
    def calculate_metric_for_item(self, item):
        """Calculate metric score for a single data sample."""
        return {}, 0.0
    
    # def get_dataset_answer(self, data):
    #     if not data.choices:
    #         return [data.golden_answers]

    #     all_choices_list = data.choices
    #     golden_choice_idx_list = data.golden_answers

    #     resposne = [
    #         [choices[idx] for idx in idx_list]
    #         for choices, idx_list in zip(all_choices_list, golden_choice_idx_list)
    #     ]
    #     return resposne
    
    def get_dataset_answer(self, data):
        # golden_answers_list = data.golden_answers
        # return golden_answers_list
        if any(choice == [] for choice in data.choices):
            golden_answers_list = data.golden_answers
        else:
            # multi-choice dataset
            all_choices_list = data.choices
            golden_choice_idx_list = data.golden_answers
            golden_answers_list = [
                [choices[idx] for idx in idx_list]
                for choices, idx_list in zip(all_choices_list, golden_choice_idx_list)
            ]

        return golden_answers_list
    
    
class F1_Score(BaseMetric):
    """Token-level F1 score"""

    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)

    def token_level_scores(self, prediction: str, ground_truths: list):
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)

            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            if (
                normalized_ground_truth in ["yes", "no", "noanswer"]
                and normalized_prediction != normalized_ground_truth
            ):
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["f1", "precision", "recall"]:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric
    
    def calculate_metric_for_item(self, item):
        pred = item.pred[0]
        golden_answers = self.get_dataset_answer(item)
        metric_result = self.token_level_scores(pred, golden_answers)
        return metric_result, metric_result["f1"]

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["f1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        f1 = sum(metric_score_list) / len(metric_score_list)
        return {"f1": f1}, metric_score_list



class Recall_Score(F1_Score):
    """Token-level Recall score"""
    metric_name = "recall"

    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric_for_item(self, item):
        return super().calculate_metric_for_item(item)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["recall"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"recall": precision}, metric_score_list


class Precision_Score(F1_Score):
    """Token-level Precision score"""

    metric_name = "precision"

    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric_for_item(self, item):
            return super().calculate_metric_for_item(item)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["precision"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"precision": precision}, metric_score_list


class ExactMatch(BaseMetric):
    r"""Exact match measure whether the predicted answer is completely consistent
    with the standard answer.

    """
    metric_name = "em"
    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"

    def calculate_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0

        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.fullmatch(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1.0
                    break
        return score
    
    def calculate_metric_for_item(self, item):
        pred = item.pred[0]
        golden_answers = self.get_dataset_answer(item)[0]
        metric_result = {"em": self.calculate_em(pred, golden_answers)}
        return metric_result, metric_result["em"]

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        em_score = sum(metric_score_list) / len(metric_score_list)

        return {"em": em_score}, metric_score_list

class Sub_ExactMatch(BaseMetric):
    r"""Sub-Exact match measure whether the predicted answer contains the standard answer."""

    metric_name = "acc"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"

    def calculate_sub_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.search(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer in normalized_prediction:
                    score = 1.0
                    break
        return score
    
    def calculate_metric_for_item(self, item):
        pred = item.pred[0]
        golden_answers = self.get_dataset_answer(item)[0]
        metric_result = {"acc": self.calculate_sub_em(pred, golden_answers)}
        return metric_result, metric_result["acc"]

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_sub_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        sub_em_score = sum(metric_score_list) / len(metric_score_list)

        return {"acc": sub_em_score}, metric_score_list
