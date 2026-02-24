import os
import json
import random
import warnings
import datasets
from typing import List, Dict, Any, Optional, Generator
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

        # if len(self.golden_answers) == 0 and "answer" in item_dict and len(self.choices)>0:
        #     self.golden_answers = [int(item_dict["answer"])]

        self.data: Dict[str, Any] = item_dict

    def update_output(self, key: str, value: Any) -> None:
        """Update the output dict and keep a key in self.output can be used as an attribute."""
        if key in ["id", "question", "golden_answers", "output", "choices"]:
            raise AttributeError(f"{key} should not be changed")
        else:
            self.output[key] = value
    
    def update_metrics(self, key: str, value: Any) -> None:
        """Update the metrics dict and keep a key in self.metrics can be used as an attribute."""
        self.metrics[key] = value

    def update_evaluation_score(self, metric_name: str, metric_score: float) -> None:
        """Update the evaluation score of this sample for a metric."""
        if "metric_score" not in self.output:
            self.output["metric_score"] = {}
        self.output["metric_score"][metric_name] = metric_score

    def __getattr__(self, attr_name: str) -> Any:
        predefined_attrs = ["id", "question", "golden_answers", "metadata", "output", "choices", "metrics", "data"]
        if attr_name in predefined_attrs:
            return super().__getattribute__(attr_name)
        else:
            output = self.output
            if attr_name in output:
                return output[attr_name]
            else:
                try:
                    return self.data[attr_name]
                except AttributeError:
                    raise AttributeError(f"Attribute `{attr_name}` not found")

    def __setattr__(self, attr_name: str, value: Any) -> None:
        predefined_attrs = ["id", "question", "golden_answers", "metadata", "output", "choices", "metrics", 'data']
        if attr_name in predefined_attrs:
            super().__setattr__(attr_name, value)
        else:
            self.update_metrics(attr_name, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all information within the data sample into a dict. Information generated
        during the inference will be saved into output field.
        """
        output = {
            "id": self.id,
            "question": self.question,
            "golden_answers": self.golden_answers,
            "choices": self.choices,
            "metadata": self.metadata,
            "output": self.output,
            "metrics": self.metrics,
        }
        return output

    def __str__(self) -> str:
        """Return a string representation of the item with its main attributes."""
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)


class Dataset:
    """A container class used to store the whole dataset. Inside the class, each data sample will be stored
    in `Item` class. The properties of the dataset represent the list of attributes corresponding to each item in the dataset.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        dataset_path: Optional[str] = None,
        data: Optional[List[Dict[str, Any]]] = None,
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
            print("Load data from provided data")
            if isinstance(data[0], dict):
                self.data = [Item(item_dict) for item_dict in data]
            else:
                assert isinstance(data[0], Item)
                self.data = data

    def _load_data(self, dataset_name: str, dataset_path: str) -> List[Item]:
        """Load data from the provided dataset_path or directly download the file(TODO)."""
        if not os.path.exists(dataset_path):
            # TODO: auto download: self._download(self.dataset_name, dataset_path)
            raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

        data = []
        if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    item_dict = json.loads(line)
                    item = Item(item_dict)
                    data.append(item)
        elif dataset_path.endswith('parquet'):
            hf_data = datasets.load_dataset('parquet', data_files=dataset_path, split="train")
            hf_data = hf_data.cast_column('image', datasets.Image())
            for item in hf_data:
                item = Item(item)
                data.append(item)
        else:
            raise NotImplementedError
        
        if self.sample_num is not None:
            self.sample_num = int(self.sample_num)
            if self.random_sample:
                print(f"Random sample {self.sample_num} items in test set.")
                data = random.sample(data, self.sample_num)
            else:
                data = data[: self.sample_num]

        return data

    def update_output(self, key: str, value_list: List[Any]) -> None:
        """Update the overall output field for each sample in the dataset."""
        assert len(self.data) == len(value_list)
        for item, value in zip(self.data, value_list):
            item.update_output(key, value)
            
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
        """Get an attribute of dataset items in batch."""
        for i in range(0, len(self.data), batch_size):
            batch_items = self.data[i : i + batch_size]
            yield [item[attr_name] for item in batch_items]

    def __getattr__(self, attr_name: str) -> List[Any]:
        return [item.__getattr__(attr_name) for item in self.data]

    def get_attr_data(self, attr_name: str) -> List[Any]:
        """For the attributes constructed later (not implemented using property),
        obtain a list of this attribute in the entire dataset.
        """
        return [item[attr_name] for item in self.data]

    def __getitem__(self, index: int) -> Item:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def save(self, save_path: str) -> None:
        """Save the dataset into the original format."""

        save_data = [item.to_dict() for item in self.data]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)


    def __str__(self) -> str:
        """Return a string representation of the dataset with a summary of items."""
        return f"Dataset '{self.dataset_name}' with {len(self)} items"
    


def convert_numpy(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: convert_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy(element) for element in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer,)):
        return int(data)
    elif isinstance(data, (np.floating,)):
        return float(data)
    elif isinstance(data, (np.bool_)):
        return bool(data)
    elif isinstance(data, (np.str_)):
        return str(data)
    else:
        return data

def filter_dataset(dataset: Dataset, filter_func=None):
    if filter_func is None:
        return dataset
    data = dataset.data
    for item in data:
        if not filter_func(item):
            data.remove(item)
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
    dataset_split_iter = {symbol: iter(dataset.data) for symbol, dataset in dataset_split.items()}

    final_data = []
    for item_symbol in split_symbol:
        final_data.append(next(dataset_split_iter[item_symbol]))
    final_dataset = Dataset(config=list(dataset_split.values())[0].config, data=final_data)

    return final_dataset


def get_batch_dataset(dataset: Dataset, batch_size=16):
    data = dataset.data
    for idx in range(0, len(data), batch_size):
        batched_data = data[idx : idx + batch_size]
        batch_dataset = Dataset(config=dataset.config, data=batched_data)
        yield batch_dataset


def merge_batch_dataset(dataset_list: Dataset):
    dataset = dataset_list[0]
    total_data = []
    for batch_dataset in dataset_list:
        total_data.extend(batch_dataset.data)
    dataset = Dataset(config=dataset.config, data=total_data)
    return dataset


