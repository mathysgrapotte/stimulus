import polars as pl
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="Process Titanic dataset to stimulus format")
    parser.add_argument("--input", type=str, help="Path to input csv file, should be identical to Kaggle download of the Titanic dataset, see : https://www.kaggle.com/c/titanic/data", required=True)
    parser.add_argument("--output", type=str, help="Path to output csv file", default="titanic_stimulus.csv", required=False)
    return parser.parse_args()  

def main():
    args = arg_parser()
    df = pl.read_csv(args.input)
    df = df.select([
        "PassengerId",
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ])

    df = df.drop_nulls()

    # Rename columns to match stimulus format

    df = df.rename({"Survived": "survived:label:int",
                "Pclass": "pclass:input:int",
                "Sex": "sex:input:str",
                "Age": "age:input:int",
                "SibSp": "sibsp:input:int",
                "Parch": "parch:input:int",
                "Fare": "fare:input:float",
                "Embarked": "embarked:input:str",
                "PassengerId": "passenger_id:meta:int"
                })

    # Save to csv
    df.write_csv(args.output)

if __name__ == "__main__":
    main()