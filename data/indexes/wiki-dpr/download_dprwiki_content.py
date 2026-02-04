import csv
import gzip
import os
from pathlib import Path
import shutil
from typing import Dict, List
import urllib.request
from tqdm import tqdm

#Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)
# "https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py"

DPR_WIKI_DOWNLOAD_URL = (
    "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
)


class PassageCollection(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.passages = []


class DPRWikiCollection(PassageCollection):
    def __init__(
        self,
        name: str = "dpr_wiki",
        file_name: str = "psgs_w100.tsv",
        cachedir: str = "data/indexes/wiki-dpr",
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = "wiki:",
        normalize: bool = True,
    ):
        super().__init__(name)
        self.id_col = id_col
        self.text_col = text_col
        self.title_col = title_col
        self.id_prefix = id_prefix
        self.normalize = normalize
        self._id_to_index = {}
        self.header_included = False
        self.load_data(cachedir + "/" + file_name)

    def load_data(self, path_to_file: str):

        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        if not os.path.exists(path_to_file):
            wget(DPR_WIKI_DOWNLOAD_URL, path_to_file, compressed=True)

        with open(path_to_file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for i, row in enumerate(tqdm(reader, desc="Loading DPR Wiki")):
                if row[self.id_col] == "id":
                    self.header_included = True
                    continue
                sample_id = self.id_prefix + str(row[self.id_col])
                passage = row[self.text_col]
                title = row[self.title_col]
                sub_title = ""
                if self.normalize:
                    passage = normalize_passage(passage)
                index = i - 1 if self.header_included else i
                self.passages.append(
                    {
                        "id": sample_id,
                        "text": passage,
                        "title": title,
                        "sub_title": sub_title,
                        "index": index,
                    }
                )
                self._id_to_index[sample_id] = index

    def get_passage_from_id(self, id: str) -> Dict[str, str]:
        passage = self.passages[self._id_to_index[id]]
        assert passage["index"] == self._id_to_index[id]
        return passage

    def get_indices_from_ids(self, ids: List[str]) -> List[int]:
        return [self._id_to_index[id] for id in ids]


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    return ctx_text


def wget(url, path, progress=True, overwrite=False, create_dir=True, compressed=False):
    """
    Download a file from a URL to a given path.

    Parameters
    ----------
    url : str
        The URL to download from.
    path : str
        The path to save the downloaded file to.
    progress : bool, optional
        Whether to display a progress bar, by default True
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, by default False
    create_dir : bool, optional
        Whether to create the directory if it doesn't exist, by default True
    compressed : bool, optional
        Whether the downloaded file is compressed, by default False
        Only works for .gz files.
    """
    if not overwrite and Path(path).exists():
        return None
    
    path = Path(path)

    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Give a nice description for the download progress bar
    if compressed:
        download_path = Path(path.as_posix() + ".gz")
    else:
        download_path = path

    if not download_path.exists():
        if progress:
            print(f"Downloading '{download_path}' from {url}")

        # Get content length of file to download
        with urllib.request.urlopen(url) as u:
            meta = u.info()
            file_size = int(meta["Content-Length"])

        # Use tqdm to display download progress, urlretrieve to download
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=file_size,
            desc=download_path.name,
            disable=not progress,
        ) as t:
            out = urllib.request.urlretrieve(
                url,
                filename=download_path,
                reporthook=lambda b, bsize, tsize: t.update(bsize),
            )

    # Decompress the file if necessary
    if compressed:
        print(f"Decompressing '{download_path}' to '{path}'")
        with gzip.open(download_path, "rb") as f_in:
            with open(path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the compressed file
        download_path.unlink()


if __name__ == "__main__":
    wiki_collection = DPRWikiCollection()
    print(f"Loaded {len(wiki_collection.passages)} passages.")