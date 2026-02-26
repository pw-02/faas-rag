# pwrag/dataset/dataset.py

import os
import json
import random
import datasets
from typing import List, Dict, Any, Optional, Generator, Iterable, Union, Iterator
import numpy as np
from pwrag.args.args import AppConfig


class Item:
    """A container class used to store and manipulate a sample within a dataset.
    Information related to this sample during training/inference will be stored in `self.output`.
    Each attribute of this class can be used like a dict key (also for key in `self.output`).
    """

    def __init__(self, item_dict: Dict[str, Any]) -> None:
        self.id: Optional[str] = item_dict.get("id", None)
        self.question: Optional[str] = item_dict.get("question", None)
        self.golden_answers: List[str] = item_dict.get("golden_answers", [])
        self.choices: List[str] = item_dict.get("choices", [])
        self.metadata: Dict[str, Any] = item_dict.get("metadata", {})
        self.output: Dict[str, Any] = item_dict.get("output", {})
        self.metrics: Dict[str, Any] = item_dict.get("metrics", {})
        self.data: Dict[str, Any] = item_dict

    # ✅ allow item["question"] style access used elsewhere in your code
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def update_output(self, key: str, value: Any) -> None:
        if key in ["id", "question", "golden_answers", "output", "choices"]:
            raise AttributeError(f"{key} should not be changed")
        self.output[key] = value

    def update_metrics(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def update_evaluation_score(self, metric_name: str, metric_score: float) -> None:
        if "metric_score" not in self.output:
            self.output["metric_score"] = {}
        self.output["metric_score"][metric_name] = metric_score

    def __getattr__(self, attr_name: str) -> Any:
        predefined_attrs = ["id", "question", "golden_answers", "metadata", "output", "choices", "metrics", "data"]
        if attr_name in predefined_attrs:
            return super().__getattribute__(attr_name)

        if attr_name in self.output:
            return self.output[attr_name]

        if attr_name in self.data:
            return self.data[attr_name]

        raise AttributeError(f"Attribute `{attr_name}` not found")

    def __setattr__(self, attr_name: str, value: Any) -> None:
        predefined_attrs = ["id", "question", "golden_answers", "metadata", "output", "choices", "metrics", "data"]
        if attr_name in predefined_attrs:
            super().__setattr__(attr_name, value)
        else:
            # keep your behavior: unknown attrs go to metrics
            self.update_metrics(attr_name, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "golden_answers": self.golden_answers,
            "choices": self.choices,
            "metadata": self.metadata,
            "output": self.output,
            "metrics": self.metrics,
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)


class Dataset:
    """A container class used to store the whole dataset."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        dataset_path: Optional[str] = None,
        data: Optional[Union[List[Dict[str, Any]], List[Item]]] = None,
        sample_num: Optional[int] = None,
        random_sample: bool = False,
    ) -> None:
        if config is not None:
            self.config = config
            dataset_name = config.dataset.dataset_name
            dataset_path = config.dataset.dataset_path
            sample_num = config.max_sample_num
            random_sample = config.random_sample
        else:
            self.config = None
            dataset_name = "default_dataset"

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.sample_num = sample_num
        self.random_sample = random_sample

        if data is None:
            self.data = self._load_data(self.dataset_name, self.dataset_path)
        else:
            # accept list[dict] or list[Item]
            if len(data) == 0:
                self.data = []
            elif isinstance(data[0], dict):
                self.data = [Item(item_dict) for item_dict in data]  # type: ignore[arg-type]
            else:
                assert isinstance(data[0], Item)
                self.data = data  # type: ignore[assignment]

    def _load_data(self, dataset_name: str, dataset_path: str) -> List[Item]:
        if dataset_path is None or not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

        data: List[Item] = []
        if dataset_path.endswith(".jsonl"):
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item_dict = json.loads(line)
                    data.append(Item(item_dict))
        elif dataset_path.endswith(".json"):
            with open(dataset_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # supports either list-of-dicts json or single-dict
            if isinstance(obj, list):
                data = [Item(x) for x in obj]
            else:
                raise ValueError("JSON dataset must be a list of items.")
        elif dataset_path.endswith("parquet"):
            hf_data = datasets.load_dataset("parquet", data_files=dataset_path, split="train")
            if "image" in hf_data.column_names:
                hf_data = hf_data.cast_column("image", datasets.Image())
            for item in hf_data:
                data.append(Item(item))
        else:
            raise NotImplementedError(f"Unsupported dataset format: {dataset_path}")

        if self.sample_num is not None:
            n = int(self.sample_num)
            if self.random_sample:
                data = random.sample(data, min(n, len(data)))
            else:
                data = data[:n]

        return data

    # ✅ easy iteration over items
    def __iter__(self) -> Iterator[Item]:
        return iter(self.data)
    
    def num_batches(self, batch_size: int) -> int:
        return (len(self.data) + batch_size - 1) // batch_size

    # ✅ iterate over dataset in batches (returns Dataset objects)
    def iter_batches(self, batch_size: int) -> Iterator["Dataset"]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        for i in range(0, len(self.data), batch_size):
            yield Dataset(config=self.config, data=self.data[i : i + batch_size])

    def update_output(self, key: str, value_list: List[Any]) -> None:
        assert len(self.data) == len(value_list)
        for item, value in zip(self.data, value_list):
            item.update_output(key, value)

    def update_metrics(self, key: str, value_list: List[Any]) -> None:
        assert len(self.data) == len(value_list)
        for item, value in zip(self.data, value_list):
            item.update_metrics(key, value)

    @property
    def question(self) -> List[Optional[str]]:
        return [item.question for item in self.data]

    @property
    def golden_answers(self) -> List[List[str]]:
        return [item.golden_answers for item in self.data]

    @property
    def id(self) -> List[Optional[str]]:
        return [item.id for item in self.data]

    @property
    def output(self) -> List[Dict[str, Any]]:
        return [item.output for item in self.data]

    @property
    def metrics(self) -> List[Dict[str, Any]]:
        return [item.metrics for item in self.data]

    def get_batch_data(self, attr_name: str, batch_size: int) -> Generator[List[Any], None, None]:
        for i in range(0, len(self.data), batch_size):
            batch_items = self.data[i : i + batch_size]
            yield [item[attr_name] for item in batch_items]  # now works via Item.__getitem__

    def __getattr__(self, attr_name: str) -> List[Any]:
        return [getattr(item, attr_name) for item in self.data]

    def get_attr_data(self, attr_name: str) -> List[Any]:
        return [item[attr_name] for item in self.data]  # now works via Item.__getitem__

    def __getitem__(self, index: int) -> Item:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def save(self, save_path: str) -> None:
        save_data = [item.to_dict() for item in self.data]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)

    def __str__(self) -> str:
        return f"Dataset '{self.dataset_name}' with {len(self)} items"


def convert_numpy(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: convert_numpy(value) for key, value in data.items()}
    if isinstance(data, list):
        return [convert_numpy(element) for element in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        return float(data)
    if isinstance(data, (np.bool_,)):
        return bool(data)
    if isinstance(data, (np.str_,)):
        return str(data)
    return data


def filter_dataset(dataset: Dataset, filter_func=None) -> Dataset:
    if filter_func is None:
        return dataset
    # ✅ do NOT mutate while iterating
    data = [item for item in dataset.data if filter_func(item)]
    return Dataset(config=dataset.config, data=data)


def split_dataset(dataset: Dataset, split_symbol: list):
    assert len(split_symbol) == len(dataset)

    data = dataset.data
    data_split = {symbol: [] for symbol in set(split_symbol)}
    for symbol in set(split_symbol):
        symbol_data = [x for x, x_symbol in zip(data, split_symbol) if x_symbol == symbol]
        data_split[symbol] = Dataset(config=dataset.config, data=symbol_data)

    return data_split


def merge_dataset(dataset_split: dict, split_symbol: list):
    assert len(split_symbol) == sum([len(data) for data in dataset_split.values()])
    dataset_split_iter = {symbol: iter(ds.data) for symbol, ds in dataset_split.items()}

    final_data = []
    for item_symbol in split_symbol:
        final_data.append(next(dataset_split_iter[item_symbol]))
    final_dataset = Dataset(config=list(dataset_split.values())[0].config, data=final_data)

    return final_dataset


def get_batch_dataset(dataset: Dataset, batch_size=16):
    # kept for backward compatibility; prefer dataset.iter_batches()
    yield from dataset.iter_batches(batch_size)


def merge_batch_dataset(dataset_list: List[Dataset]) -> Dataset:
    base = dataset_list[0]
    total_data = []
    for batch_dataset in dataset_list:
        total_data.extend(batch_dataset.data)
    return Dataset(config=base.config, data=total_data)