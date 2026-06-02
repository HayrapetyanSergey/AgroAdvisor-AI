import pandas as pd

FILE_PATH = "/home/sergey/Desktop/LLM_Agro/data/Agriculture_Data_YSU_18.04.25.xlsx"

df = pd.read_excel(FILE_PATH)

df["Մարզ"] = df["Մարզ"].astype(str).str.strip()

# Clean column names
df.columns = [str(col).strip() for col in df.columns]


def get_dataset_info():
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "regions": sorted(df["Մարզ"].unique().tolist()),
        "years": sorted(df["Տարի"].unique().tolist())
    }


def search_columns(keyword):
    keyword = keyword.lower().strip()
    return [col for col in df.columns if keyword in col.lower()]


def get_region_data(region):
    region = region.strip()
    result = df[df["Մարզ"] == region]
    return result


def get_average_by_region(column_name, exclude_total=True):
    data = df.copy()

    if exclude_total:
        data = data[data["Մարզ"] != "Հայաստանի Հանրապետություն, ընդամենը"]

    if column_name not in data.columns:
        return None

    data[column_name] = pd.to_numeric(
        data[column_name],
        errors="coerce"
    )

    data = data.dropna(subset=[column_name])

    result = (
        data.groupby("Մարզ")[column_name]
        .mean()
        .sort_values(ascending=False)
        .round(2)
        .reset_index()
    )

    return result


def get_top_regions_by_yield(crop_keyword, top_n=5):
    columns = search_columns(crop_keyword)

    yield_columns = [
        col for col in columns
        if "1 հեկտարի միջին բերքատվությունը" in col
    ]

    if not yield_columns:
        return f"No yield column found for crop keyword: {crop_keyword}"

    column_name = yield_columns[0]

    result = get_average_by_region(column_name)

    return {
        "crop_keyword": crop_keyword,
        "used_column": column_name,
        "top_regions": result.head(top_n)
    }


def get_climate_average_by_region():
    climate_cols = [
        "Ջրտուք",
        "օդի միջին հարաբերական խոնավություն(%)",
        "մթնոլորտային ճնշում(հՊա)",
        "տեղումների քանակ(մմ)",
        "օդի միջին ջերմաստիճան(°C)"
    ]

    data = df[df["Մարզ"] != "Հայաստանի Հանրապետություն, ընդամենը"]

    result = (
        data.groupby("Մարզ")[climate_cols]
        .mean()
        .round(2)
        .reset_index()
    )

    return result


def get_trend_by_region(region, column_name):
    region = region.strip()

    if column_name not in df.columns:
        return None

    result = (
        df[df["Մարզ"] == region][["Տարի", column_name]]
        .sort_values("Տարի")
        .reset_index(drop=True)
    )

    return result


if __name__ == "__main__":
    print("\nDATASET INFO:")
    print(get_dataset_info())

    print("\nTOP REGIONS BY GRAPE YIELD:")
    grape = get_top_regions_by_yield("խաղող")
    print(grape["used_column"])
    print(grape["top_regions"])

    print("\nTOP REGIONS BY POTATO YIELD:")
    potato = get_top_regions_by_yield("կարտոֆիլ")
    print(potato["used_column"])
    print(potato["top_regions"])

    print("\nCLIMATE AVERAGE BY REGION:")
    print(get_climate_average_by_region())

grape_col = search_columns("խաղող")[2]

trend = (
    df[df["Մարզ"] == "Արարատ"]
    [["Տարի", grape_col]]
    .sort_values("Տարի")
)

print(trend)