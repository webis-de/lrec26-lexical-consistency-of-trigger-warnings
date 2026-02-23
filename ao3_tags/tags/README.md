# Tags package
The code in this package collects tags that belong to one root tag. Examples of a root tag are `Violence` or `Abuse`.

## Tag graph
The script [`run_spark.sh`](run_spark.sh) submits a job to the Kubernetes-based Spark cluster to parse a graph of tags.
- It uses the JSON-L-files in the `tag_path` of [config.yaml](../conf/config.yaml) to construct the graph
- For a specified root node, the job collects all tags that are either sub tags or synonymous tags

The job can be run as follows:
```
./run_spark [ROOT_NODE] [MIN_DISTANCE] [MAX_DISTANCE]
```
- `ROOT_NODE` is the node from which to start the graph traversal. Names are case sensitive
- `MIN_DISTANCE` defines the minimum number of edges that a node needs to be away from the root node
- `MAX_DISTANCE` defines the maximum number of edges that a node can be away from the root node

Example
```
./run_spark Abuse 1 10
```
The results are stored in the output directory under [./output/data/tags/](../../../output).

### Output
The Spark job creates a set of parquet files with three columns: `src`, `relationship`, and `dst`. 
- `src` is the starting node
- `dst` is the end node
- `relationship` states how the two edges are connected

The names of the files correspond to the number of edges from the `dst`-nodes to the original root node:
- `Abuse_edges.parquet-part-004-010` contains nodes that are 4 edges away from the root node `Abuse`

The postprocessing job then takes these files to create a single `.csv`-file of unique `dst`-nodes.

### Manual labeling of warning categories and functional qualifiers
Once the CSV-file from the tag graph is created, it can be used to categorize the tags. 
An example is the file [abuse.csv](../../resources/tags/abuse.csv):
- Each warning category is added as a column; If a tag belongs to that category, the row receives a `1`; if not a `0`

After the tags are labeled, the new file needs to be placed into [resources/tags/](../../resources/tags) to be referenced by downstream tasks.


### Related files
The following files are necessary for the Spark job:
- [create_image.sh](create_image.sh): Create a Docker image for the Kubernetes cluster
- [Dockerfile](Dockerfile): Dockerfile for the image
- [requirements.txt](requirements.txt): Dependencies for the Docker image
- [tag_graph.py](tag_graph.py): Pyspark script for the Spark job
- [tag_graph_postprocessing.py](tag_graph_postprocessing.py): Local script to aggregate the output of the Spark job

## AO3 Tag Page
For illustration purposes, the script [`ao3_tag_page.py`](./ao3_tag_page.py) scrapes all synonymous tags for a root tag from the corresponding page on https://archiveofourown.org/tags/[ROOT_TAG]
```
python -m ao3_tags.tags.ao3_tag_page [ROOT_TAG]
```
Example
```
python -m ao3_tags.tags.ao3_tag_page Abuse
```