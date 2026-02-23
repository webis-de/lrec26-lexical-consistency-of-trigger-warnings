import pandas as pd

from ao3_tags import DATA_PATH


def run(root_tag: str = "Abuse"):
    output_dir = DATA_PATH / "tags"
    edge_files = output_dir.glob(f"{root_tag}_edges.parquet-part-" + "[0-9]"*3 + "-" + "[0-9]"*3)
    df = pd.DataFrame()
    for file_path in edge_files:
        tmp_df = pd.read_parquet(file_path)
        tmp_df["num_edges"] = int(file_path.name.split("-")[-2])
        df = pd.concat([df, tmp_df], ignore_index=True)

    # Remove any duplicates
    df = df.drop_duplicates("dst").sort_values(["num_edges", "src", "dst"]).drop(columns="num_edges")
    df.to_csv(output_dir / f"{root_tag}_edges.csv", index=False)
    print(f"\nSaved aggregated edges for {root_tag} to {output_dir / f'{root_tag}_edges.csv'}.")


if __name__ == '__main__':
    # Parse the input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root_tag', metavar='r', type=str,
                        help="Name of the tag that forms the root of the tree. Used to load edge-files.")
    args = parser.parse_args()

    # Run the job as specified
    run(root_tag=args.root_tag)