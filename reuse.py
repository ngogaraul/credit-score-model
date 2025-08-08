from multi_customer_predictor import load_model_and_predict_for_multiple_customers
results=load_model_and_predict_for_multiple_customers(
    data_path="C:/Users/bngog/Desktop/intern/css pj/large_synthetic_test_dataset.xlsx",
    model_path="C:/Users/bngog/Desktop/intern/css pj/credit_score_model.pkl",
    scaler_path="C:/Users/bngog/Desktop/intern/css pj/credit_score_scaler.pkl",
    output_path="C:/Users/bngog/Desktop/intern/css pj/output/zigama_output3.xlsx"

)

print(results)
