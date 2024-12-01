import pandas as pd

def prepare_dateset(data_path):
    calendar = pd.read_csv(f"{data_path}/calendar.csv")
    sell_prices = pd.read_csv(f"{data_path}/sell_prices.csv")
    sales = pd.read_csv(f"{data_path}/sales_train_evaluation.csv")

    sales = sales.melt(id_vars=["id","item_id","dept_id","cat_id","store_id","state_id"],var_name = "d")
    sales["day_num"] = sales["d"].str[2:].astype(int)
    calendar = calendar.melt(id_vars = ["date", "wm_yr_wk",	"weekday", "wday", "month",	"year",	"d", "event_name_1", "event_type_1", "event_name_2", "event_type_2"],var_name="state_id",value_name = 'snap')
    calendar["state_id"] = calendar["state_id"].str[5:]
    calendar["date"] = pd.to_datetime(calendar["date"], format = '%Y-%m-%d')
    
    sales = sales.merge(calendar, on = ["d","state_id"], how = "left" )

    sales = sales.merge(sell_prices, on = ["store_id","item_id","wm_yr_wk"], how = "left")

    sales["total_volume"] = sales["sell_price"]*sales["value"]
    sales["train"] = sales["day_num"]<=1913

    return(sales)

def load_data(prepare=False):

    data_path = 'm5-forecasting-accuracy'

    if prepare:
        sales = prepare_dateset(data_path)
        sales.to_parquet(f"{data_path}/sales_train_prepared.parquet")
    else:
        sales = pd.read_parquet(f"{data_path}/sales_train_prepared.parquet")

    return(sales)

def null_summary(df):
    summary_list = []

    for col in df.columns:
        summary = (
            df
                .groupby(col)
                .agg( **{f"zero_share_{col}" : ("value", lambda x: (x == 0).sum() / len(x))} )
                .describe()
        )
        summary_list.append(summary)

    return pd.concat(summary_list, axis=1)