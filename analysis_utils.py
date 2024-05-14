import pandas as pd
from typing import Dict, List, Any
import numpy as np
from user_agents import parse


def fill_na_based_on_condition(
    df: pd.DataFrame,
    condition_col: str,
    target_cols: List[str],
    condition_val: int = 0,
    fill_val: int = -1,
) -> pd.DataFrame:
    mask = df[condition_col] == condition_val
    df.loc[mask, target_cols] = df.loc[mask, target_cols].fillna(fill_val)
    return df


def fix_inconsistent_col_format(
    df: pd.DataFrame,
    column_name: str,
    indicator_col: str,
    numeric_indicator: Any,
    string_indicator: Any,
    regex_pattern: str = r"[^\d]+",
) -> pd.DataFrame:

    mask_numeric = df[indicator_col] == numeric_indicator
    df.loc[mask_numeric, column_name] = pd.to_numeric(
        df.loc[mask_numeric, column_name], errors="coerce"
    ).astype(int)

    # Process rows where type is 'str'
    mask_str = df[indicator_col] == string_indicator
    df.loc[mask_str, column_name] = (
        df.loc[mask_str, column_name]
        .astype(str)
        .str.replace(regex_pattern, "", regex=True)
    )
    df.loc[mask_str, column_name] = pd.to_numeric(
        df.loc[mask_str, column_name], errors="coerce"
    ).astype(int)

    return df


def percentage_nans(df: pd.DataFrame) -> Dict:
    return df.isna().mean() * 100


def get_cols_type(df: pd.DataFrame) -> Dict:
    return {col: pd.api.types.infer_dtype(df[col]) for col in df.columns}


def fill_na_based_on_other_column(
    df: pd.DataFrame, target_col: str, source_col: str, prefix: str = ""
) -> pd.DataFrame:
    df[target_col] = np.where(
        pd.isna(df[target_col]) & pd.notna(df[source_col]),
        prefix + df[source_col],
        df[target_col],
    )

    return df


def get_group_mode_to_dict(df: pd.DataFrame, key_col: str, value_col: str):
    return (
        df.groupby(key_col)[value_col]
        .agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None)
        .dropna()
        .to_dict()
    )


def map_na_according_to_dict(
    df: pd.DataFrame, key_col: str, value_col: str, dict_for_mapping: Dict
):
    return df.assign(
        **{f'{value_col}':df.apply(
            lambda row: (
                dict_for_mapping.get(row[key_col], pd.NA)
                if pd.isna(row[value_col])
                else row[value_col]
            ),
            axis=1,
        )}
    )


def fillna_with_group_mode_to_dict(df:pd.DataFrame, key_col:str, value_col:str):
    group_to_mode = get_group_mode_to_dict(df, key_col, value_col)
    return map_na_according_to_dict(df, key_col, value_col, group_to_mode)


from user_agents import parse
import pandas as pd

def extract_features(user_agent):
    try:
        ua = parse(user_agent)
    except Exception as e:
        print(f"Error parsing user agent: {e}")
        return pd.Series({  # Return a series with default values in case of an error
            'browser_family': None,
            'browser_version': None,
            'os_family': None,
            'os_version': None,
            'device_family': None,
            'device_brand': None,
            'device_model': None,
            'is_mobile': False,
            'is_tablet': False,
            'is_pc': False,
            'is_touch_capable': False,
            'is_bot': False
        })

    return pd.Series({
        'browser_family': ua.browser.family if ua.browser.family else "Unknown",
        'browser_version': ua.browser.version_string if ua.browser.version_string else "Unknown",
        'os_family': ua.os.family if ua.os.family else "Unknown",
        'os_version': ua.os.version_string if ua.os.version_string else "Unknown",
        'device_family': ua.device.family if ua.device.family else "Unknown",
        'device_brand': ua.device.brand if ua.device.brand else "Unknown",
        'device_model': ua.device.model if ua.device.model else "Unknown",
        'is_mobile': ua.is_mobile,
        'is_tablet': ua.is_tablet,
        'is_pc': ua.is_pc,
        'is_touch_capable': ua.is_touch_capable,
        'is_bot': ua.is_bot
    })
