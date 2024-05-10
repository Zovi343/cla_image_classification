from training import training
from inference import inference
from evaluation import compute_f1score4_classes

if __name__ == "__main__":
    path_2_dataset = "../public/data_cla_public"
    train_model = False

    if train_model:
        training(path_2_dataset)

    path_2_model = './final_files/final_model.pt'
    n_samples_2_predict = -1

    inference(path_2_dataset, path_2_model, n_samples_2_predict)

    project_type = "CLA"
    path_2_ground_truth = 'ground_truth'
    path_2_predictions = 'output_predictions'

    score = compute_f1score4_classes(project_type, path_2_ground_truth, path_2_predictions)

    print(f'Final score for {project_type} project: {score}')