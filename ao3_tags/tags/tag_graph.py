from graphframes import GraphFrame
import json
from pyspark import SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, MapType, ArrayType

from ao3_tags import DATA_PATH, TAG_PATH
DATA_PATH_STR = str(DATA_PATH)
TAG_PATH_STR = str(TAG_PATH)

MAIN_SCHEMA = StructType([
    StructField("tag_name",
                StringType(),
                True),
    StructField("tag_url",
                StringType(),
                True),
    StructField("tag_type",
                StringType(),
                True),
    StructField("is_common",
                BooleanType(),
                True),
    StructField("canonical_tag",
                MapType(StringType(), StringType(), True),
                True),
    StructField("parent_tags",
                ArrayType(MapType(StringType(), StringType(), True), True),
                True),
    StructField("child_tags",
                MapType(
                    StringType(), ArrayType(MapType(StringType(), StringType(), True), True)
                ),
                True),
    StructField("synonymous_tags",
                ArrayType(MapType(StringType(), StringType(), True), True),
                True),
    StructField("meta_tags",
                ArrayType(MapType(StringType(), StringType(), True), True),
                True),
    StructField("sub_tags",
                ArrayType(MapType(StringType(), StringType(), True), True),
                True),
    StructField("used_by_works",
                ArrayType(StringType(), True),
                True),
    StructField("used_by_bookmarks",
                ArrayType(StringType(), True),
                True)
])

SUB_SCHEMA = StructType([
    StructField("tag_name",
                StringType(),
                True),
    StructField("synonymous_tags",
                ArrayType(StringType(), True),
                True),
    StructField("sub_tags",
                ArrayType(StringType(), True),
                True)
])

EDGE_SCHEMA = StructType([
    StructField('src',
                StringType(),
                True),
    StructField('relationship',
                StringType(),
                True),
    StructField('dst',
                StringType(),
                True)
])


def parse_json(line):
    try:
        return [Row(**json.loads(line))]
    except json.decoder.JSONDecodeError:
        return []


def create_edges(df, edge_type):
    return df.selectExpr("tag_name as src", f"explode({edge_type}) as dst").withColumn("relationship", lit(edge_type))


def generate_motif(num_edges=1):
    motif_list = []
    for i in range(1, num_edges+1):
        src = f"v{i}"
        if i == 1:
            src = "a"
        motif_list.append(f"({src})-[e{i}]->(v{i+1})")

    return "; ".join(motif_list)


def run(root_tag: str = "Abuse", min_distance: int = 1, max_distance: int = 10,
        data_path: str = DATA_PATH_STR, tag_path: str = TAG_PATH_STR):
    sc = SparkContext().getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # Load the lines of the JSON-L file
    file_path = f"{tag_path}/tags_info.jsonl"
    lines = sc.textFile(file_path)

    # Parse the JSON, extract the names from the tag lists and keep only the relevant columns
    json_data = lines.flatMap(parse_json)
    df = spark.createDataFrame(json_data, schema=MAIN_SCHEMA)

    rdd = df.rdd.map(lambda x: {"tag_name": x["tag_name"],
                                "synonymous_tags": [y['name'] for y in x["synonymous_tags"]],
                                "sub_tags": [z['name'] for z in x["sub_tags"]]}
                     )
    df2 = rdd.toDF(schema=SUB_SCHEMA)

    # Create the edges and vertices and from them the graph of tags
    edges_syn = create_edges(df2, "synonymous_tags").distinct()
    edges_sub = create_edges(df2, "sub_tags").distinct()
    edges = edges_syn.unionByName(edges_sub).distinct()

    vertices = df2.selectExpr("tag_name as id")
    g = GraphFrame(vertices, edges)


    # Iteratively look for nodes with a given number of edges to the root node
    out_prefix = f"{data_path}/tags/{root_tag}_edges"
    i = min_distance
    while i <= max_distance:
        pattern = generate_motif(num_edges=i)
        motifs = g.find(pattern)

        # Filter for all motifs that have the root_tag as starting point. If no results were found, stop the iteration
        df_result = motifs.filter(f"a.id == '{root_tag}'")
        if len(df_result.head(1)) == 0:
            break

        # Get only the information from the last edge column as the rest is already covered by previous iterations
        df_e = (df_result.rdd.map(lambda x: (x[f"e{i}"]["src"],
                                             x[f"e{i}"]["relationship"],
                                             x[f"e{i}"]["dst"]))
                .toDF(["src", "relationship", "dst"]))

        # Write the results and update i
        df_e.repartition(20).write.parquet(f"{out_prefix}.parquet-part-{i:03}-{max_distance:03}", mode='overwrite')
        i += 1


if __name__ == '__main__':
    # Parse the input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root_tag', metavar='r', type=str,
                        help="Name of the tag that forms the root of the tree. Other tags will be collected from it.")
    parser.add_argument('min_distance', metavar='m', type=int,
                        help="Minimum number of edges a node has to be away from the root.")
    parser.add_argument('max_distance', metavar='m', type=int,
                        help="Maximum number of edges a node can be away from the root.")
    parser.add_argument('data_path', metavar='d', type=str,
                        help="Path that contains the project's data")
    parser.add_argument('tag_path', metavar='t', type=str,
                        help="Path that contains the JSON-L file with the tag information")
    args = parser.parse_args()

    # Run the job as specified
    run(root_tag=args.root_tag,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        data_path=args.data_path,
        tag_path=args.tag_path)
