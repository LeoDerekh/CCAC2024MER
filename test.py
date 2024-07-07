import os
import time
import zipfile

import numpy as np
import torch
from tqdm import tqdm

from ema import EMA
from model.model_utils import get_model
from option.option import Options
from utils.utils import load_me_data, evaluate, get_test_loader


def validate_epoch_model(epoch_model_path, model, criterion, val_loader, opt, ema):
    epoch_model = torch.load(epoch_model_path, map_location=opt.device)
    epoch = epoch_model["epoch"]
    model.load_state_dict(epoch_model["state_dict"])
    # 加载并应用 EMA 参数
    ema.shadow = epoch_model.get('ema_shadow', {})
    ema.apply_shadow()
    val_loss, val_UF1, val_UAR, val_ACC, val_class_accuracies = evaluate(
        opt=opt,
        model=model,
        criterion=criterion,
        data_loader=val_loader,
        device=opt.device,
        epoch=epoch,
    )
    print(
        "Validation Epoch {} => val_UF1: {:.4f}, val_UAR: {:.4f}, val_ACC: {:.4f}, val_class_accuracies: {}".format(
            epoch, val_UF1, val_UAR, val_ACC, val_class_accuracies
        )
    )
    print()
    # 恢复原始参数
    ema.restore()
    return val_UF1, epoch_model_path


# Function to get predictions from a model
def get_predictions(model_path, model, test_loader, opt, ema):
    epoch_model = torch.load(model_path, map_location=opt.device)
    model.load_state_dict(epoch_model["state_dict"])
    # 加载并应用 EMA 参数
    ema.shadow = epoch_model.get('ema_shadow', {})
    ema.apply_shadow()
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for inputs in tqdm(test_loader):
            if opt.input_type == "apex_flow":
                apex = inputs[0].to(opt.device)
                optical_flow = inputs[1].to(opt.device)
                outputs = model(apex, optical_flow)
            else:
                inputs = inputs.to(opt.device)
                outputs = model(inputs)
        

            _, preds = torch.max(outputs, 1)
            all_predictions.append(preds.cpu().numpy())
    # 恢复原始参数
    ema.restore()
    return np.concatenate(all_predictions)


def majority_vote_with_tie_break(x):
    """Performs majority voting and breaks ties by selecting the first model's prediction."""
    votes = np.bincount(x)
    max_count = np.max(votes)
    if np.sum(votes == max_count) > 1:  # There's a tie
        return x[0]  # Select the first model's prediction in case of a tie
    else:
        return votes.argmax()


def test(opt, best_model_path):

    train_loader, val_loader = load_me_data(opt)
    model = get_model(opt)
    # 实例化 EMA
    ema = EMA(model, decay=opt.ema_decay)

    # Load the main model and determine the current epoch
    best_epoch_model = torch.load(best_model_path, map_location=opt.device)
    best_epoch = best_epoch_model["epoch"]

    criterionCE = torch.nn.CrossEntropyLoss().to(opt.device)
    torch.nn.DataParallel(criterionCE, opt.gpu_ids)

    # Validate models and collect val_UF1 scores
    val_scores = []
    k = 5
    for i in range(best_epoch - k, best_epoch + k + 1):
        epoch_model_path = best_model_path.replace(
            "best_model", "model_{}".format(i))
        if not os.path.exists(epoch_model_path):
            continue
        val_UF1, model_path = validate_epoch_model(
            epoch_model_path, model, criterionCE, val_loader, opt, ema
        )
        val_scores.append((val_UF1, model_path))

    # Sort models by val_UF1 and select the top 3
    val_scores.sort(reverse=True, key=lambda x: x[0])
    top_models = val_scores[:3]
    sum_of_values = sum([item[0] for item in top_models])
    average = sum_of_values / len(top_models)
    print("Average of top 3:", average)


    # Load test data loader
    test_loader = get_test_loader(opt)

    # Get predictions from the top 3 models
    predictions = [
        get_predictions(model_path, model, test_loader, opt, ema)
        for _, model_path in top_models
    ]

    # Voting mechanism
    final_predictions = np.apply_along_axis(
        majority_vote_with_tie_break, axis=0, arr=np.array(predictions))

    print(final_predictions)
    print(len(final_predictions))

    # Save final_predictions to prediction.txt
    with open("prediction.txt", "w", encoding="utf-8") as f:
        for pred in final_predictions:
            f.write(f"{int(pred)}\n")

    # Create a zip file containing prediction.txt
    with zipfile.ZipFile("prediction.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write("prediction.txt", arcname="prediction.txt")

    os.remove("prediction.txt")

    print("Prediction results have been saved and zipped.")



if __name__ == "__main__":
    opt = Options().parse()

    opt.model = "testmodel"    
    date_time = "240704_201727"
    # convnext_tiny     240704_110303     0.39662944863773336     0.9
    # testmodel     240704_110632       0.40092890161256173     0.9
    # testmodel   240704_113502      0.3926002639686699     0.999
    
    if opt.model == "dualmodel":
        opt.input_type = "apex_flow"

    test(opt, best_model_path="/NAS/xiaohang/CCAC2024MER/checkpoints/{}/{}/best_model.pth".format(opt.model, date_time))
