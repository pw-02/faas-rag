# faasrag/core/s3_utils.py
import os
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError

def is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")

def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    u = urlparse(s3_uri)
    if u.scheme != "s3" or not u.netloc or not u.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return u.netloc, u.path.lstrip("/")

def ensure_s3_file_local(
    s3_uri: str,
    local_base_dir: str,
    mirror_key_under_base: bool = True,
    skip_if_exists: bool = True,
) -> str:
    bucket, key = parse_s3_uri(s3_uri)

    local_path = (
        os.path.join(local_base_dir, key)
        if mirror_key_under_base
        else os.path.join(local_base_dir, os.path.basename(key))
    )

    if skip_if_exists and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    s3 = boto3.client("s3")
    # head_object gives a nice error if perms missing
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise FileNotFoundError(f"Cannot access {s3_uri}: {e}") from e

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    tmp = local_path + ".partial"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass

    print(f"[S3] Downloading {s3_uri} -> {local_path}")
    s3.download_file(bucket, key, tmp)
    os.replace(tmp, local_path)

    return local_path
