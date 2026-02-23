# Get Annotations for Passages
This package contains code to get annotations from an LLM using sociodemographic prompting.

## LLM in Docker image with Slurm
### Create the Docker image
From the [top directory](../../), run the following script to update the Docker image
```
./ao3_tags/annotate/create_image.sh
```

### Verify the image
If you want to verify the image locally, run the container with GPU support. This requires the NVIDIA Container Toolkit
```
# Set up the package repository and the GPG key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sed 's#deb/#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] #' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package listings and install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker to apply changes
sudo systemctl restart docker
```
Run the image
```
docker run --gpus all -v /mnt:/mnt -it <image-name> /bin/bash
```

### [llm.py](llm.py)
The main script is [llm.py](llm.py) that uses a locally hosted model to generate the annotations.
This script can be run using the [slurm_job.sh](slurm_job.sh):
```
sbatch slurm_job.sh [WARNING] [CATEGORY] [JOB_ID] [SAMPLE_PASSAGES] [N_PASSAGES]
```
- `warning`: The warning for which to get annotations
- `category`: The category for which to get annotations (Used for prompt generation)
- `job_id`: ID to identify the passages that were sampled

## [hugging_face_api.py](hugging_face_api.py)
As an alternative to the local LLM, the hugging face API can be used to generate annotations. 
This is meant primarily for illustration purposes as it is a lot slower than a local model.
```
python -m ao3_tags.annotate.hugging_face_api [WARNING] [CATEGORY] [JOB_ID]
```
- `warning`: The warning for which to get annotations
- `category`: The category for which to get annotations (Used for prompt generation)
- `job_id`: ID to identify the passages that were sampled

## Output
The output of the scripts will be created in [data/annotations/warning](../../../output/data)
- `[JOB_ID]_ids.txt`: IDs of the passages sampled for the annotations
- `[JOB_ID]_annotations.jsonl`: Annotations

An example row in the annotations files looks like this:
```
{
    "passage_id": "10003487-10-4", 
    "response": "no", 
    "gender": "female", 
    "race": "White", 
    "education": "Some college but no degree", 
    "age": "Under 18", 
    "political_affiliation": "Liberal", 
    "annotator_id": 0
}
```
Most keys relate to sociodemographic attributes. The others are
- `passage_id` is the ID of the passage shown in annotation
- `response` is the binary annotation decision (`yes` or `no`)
- `annotator_id` identifies the sociodemographic profile


## Other files
- [abstract.py](abstract.py): Abstract class for the annotator (used in LLM and Hugging Face API)
- [create_image.sh](create_image.sh): Create Docker image for the SLURM job
- [Dockerfile](Dockerfile): Dockerfile for the SLURM job
- [load_passages.py](load_passages.py): Sample passages for the annotation
- [requirements.txt](requirements.txt): Requirements for the Docker image
- [utils.py](utils.py): Create the prompts and the profiles for sociodemographic prompting
