import os
import json

from typing import List

import pandas as pd

from ..settings import ResultSetting


def read_result_meta(
    result_meta_dir: str,
    trainer_setting_path: str = None,
    subject_setting_path: str = None,
    feature_setting_path: str = None,
):
    """
    read and scan result meta files.
    """
    # scanning result meta json files and put it into a dataframe
    result_meta_files = os.listdir(result_meta_dir)
    result_meta_files = [f for f in result_meta_files if f.endswith(".json")]
    ## read json and cast it into ResultSetting
    result_meta_list = []
    for f in result_meta_files:
        with open(os.path.join(result_meta_dir, f), "r") as f:
            result_config = ResultSetting(**json.load(f))
            result_meta_list.append(result_config.dict())

    result_meta_df = pd.DataFrame(result_meta_list)

    # add result_meta_files to result_meta_df
    result_meta_df["result_meta_file"] = [
        os.path.join(result_meta_dir, f) for f in result_meta_files
    ]

    # filter by trainer_setting_path
    if trainer_setting_path:
        result_meta_df = result_meta_df[
            result_meta_df["trainer_config_path"] == trainer_setting_path
        ]

    # filter by subject_setting_path
    if subject_setting_path:
        result_meta_df = result_meta_df[
            result_meta_df["subject_config_path"] == subject_setting_path
        ]

    # filter by feature_setting_path
    if feature_setting_path:
        result_meta_df = result_meta_df[
            result_meta_df["feature_config_path"] == feature_setting_path
        ]

    result_meta_df.sort_values(
        [
            "subject_config_path",
            "trainer_config_path",
            "feature_config_path",
        ],
        inplace=True,
    )

    return result_meta_df.reset_index(drop=True)


def delete_empty_result(result_meta_df: pd.DataFrame):
    """
    delete empty result from result_meta_df
    """
    for i in range(len(result_meta_df)):
        # check id stats_path is a file
        path = result_meta_df.loc[i, "stats_path"]
        if not os.path.isfile(path):
            print(f"deleting {i}")
            try:
                result_dir = result_meta_df.loc[i, "result_dir"]
                os.system(f"rm -r {result_dir}")
            except:
                pass

            # remove meta
            try:
                meta_path = result_meta_df.loc[i, "result_meta_file"]
                os.system(f"rm {meta_path}")
            except:
                pass

            # drop from meta_df
            result_meta_df = result_meta_df.drop(i)

    return result_meta_df.reset_index(drop=True, inplace=True)


def delete_result(result_meta_df: pd.DataFrame, indices: List[int]):
    """
    delete result from result_meta_df
    """
    for idx in indices:
        print(f"deleting {idx}")
        try:
            result_dir = result_meta_df.loc[idx, "result_dir"]
            os.system(f"rm -r {result_dir}")
        except:
            pass

        # remove meta
        try:
            meta_path = result_meta_df.loc[idx, "result_meta_file"]
            os.system(f"rm {meta_path}")
        except:
            pass

        # drop from meta_df
        result_meta_df = result_meta_df.drop(idx)

    return result_meta_df.reset_index(drop=True, inplace=True)
