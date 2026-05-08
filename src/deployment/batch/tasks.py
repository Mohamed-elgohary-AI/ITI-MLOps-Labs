from prefect import task
import duckdb
import mlflow
import dagshub

@task
def load_data():
    con = duckdb.connect("md:my_db")
    return con.execute("SELECT * FROM test").df()

@task
def load_model():
    dagshub.init(
        repo_name="ITI-MLOps-Labs",
        repo_owner="mohamedabdelmonemelgohary"
    )   
    return mlflow.sklearn.load_model("models:/rf-classifier@production")

@task
def predict(model, df):
    df["prediction"] = model.predict(df)
    return df

@task
def save_results(df):
    con = duckdb.connect("md:my_db")
    con.execute("CREATE OR REPLACE TABLE predictions AS SELECT * FROM df")