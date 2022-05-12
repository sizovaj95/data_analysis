import pandas as pd

import analyse_descriptions as ad
import constants as co


def main():
    data = pd.read_csv(co.data_dir / "wine_descriptions.csv")
    ad.analyse_descriptions(data)


if __name__ == "__main__":
    main()