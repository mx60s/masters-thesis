import subprocess

url_prefix = (
    "https://s3.us-west-1.wasabisys.com/predictive-coding-thesis/datasets/"
)
with open("files.txt", "r") as f:
    for line in f:
        url = url_prefix + line.strip()
        subprocess.run(["curl", "-LO", url])