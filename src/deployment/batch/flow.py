from prefect import flow
from tasks import load_data, load_model, predict, save_results

@flow
def batch_inference_flow():
    df = load_data()
    model = load_model()
    result = predict(model, df)
    save_results(result)

if __name__ == "__main__":
    batch_inference_flow()