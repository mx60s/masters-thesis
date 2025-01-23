import subprocess

url_prefix = (
    "https://thesis-github-files.s3.us-east-2.amazonaws.com/weights/"
)
with open("files.txt", "r") as f:
    for line in f:
        url = url_prefix + line.strip()
        subprocess.run(["curl", "-LO", url])
