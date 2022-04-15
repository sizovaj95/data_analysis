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
