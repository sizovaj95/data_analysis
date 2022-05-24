from pathlib import Path
from typing import List, Tuple

wine_table_name = "winemag-data_first150k.csv"

data_dir = Path('__file__').parent.resolve().parent.resolve() / 'data'
TAGGED_WORDS = List[Tuple[str, str]]

country = "country"
description = "description"
points = "points"
price = "price"
variety = "variety"
winery = "winery"
province = "province"
points_group = "points_group"
log10_price = "log10_price"


food_labels = ["FOOD_MEAT_AND_POULTRY", "FOOD_SEAFOOD", "FOOD_GENERAL_DISHES", "FOOD_COUNTRY_CUISINE", "FOOD_FRUITS",
               "FOOD_OTHER"]

custom_nlp_model_name = 'custom_ner_2022_5_10_147236'

weird_flavors = [r"graphite", r"tobacco", r"cedar", r"asphalt", r"\boak\b", r"\btalc", r"chalk"]
