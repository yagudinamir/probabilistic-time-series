import csv
import os


def write_result(results_file, result):
    """Writes results to a csv file."""
    with open(results_file, "a+", newline="") as csvfile:
        field_names = result.keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if os.stat(results_file).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(result)


def report_baseline_results(
    model, dataset, train_frac, loss_name, seed, save, save_dir="results"
):
    result = {
        "dataset": dataset,
        "rmse": getattr(model, "rmse", 0),
        "loss": loss_name,
        "ece": getattr(model, "ece", 0),
        #              "stddev": getattr(model, "sharpness", 0),
        #              "point_unbiasedness_max": getattr(model, "point_unbiasedness_max", 0),
        #              "point_unbiasedness_mean": getattr(model, "point_unbiasedness_mean", 0),
        "threshold_calibration_error": getattr(model, "threshold_calibration_error", 0),
        #        "distribution_calibration_error": getattr(model, "distribution_calibration_error", 0),
        "point_calibration_error": getattr(model, "point_calibration_error", 0),
        "point_calibration_error_uniform_mass": getattr(
            model, "point_calibration_error_uniform_mass", 0
        ),
        #              "false_positive_rate_error": getattr(model, "false_positive_rate_error", 0),
        #              "false_negative_rate_error": getattr(model, "false_negative_rate_error", 0),
        "true_vs_pred_loss": getattr(model, "true_vs_pred_loss", 0),
        "decision_loss": getattr(model, "decision_loss", 0),
        #              "posthoc_recalibration": posthoc_recalibration,
        "test_nll": getattr(model, "test_nll", 0),
        "train_frac": train_frac,
        #              "learning_rate": learning_rate,
        "seed": seed,
    }
    results_file = save_dir + "/" + save + ".csv"
    write_result(results_file, result)

    all_err = getattr(model, "all_err", [])
    all_loss = getattr(model, "all_loss", [])
    all_y0 = getattr(model, "all_y0", [])
    all_c = getattr(model, "all_c", [])

    decision_making_results_file = (
        save_dir
        + "/"
        + save
        + "_decision_{}_{}_{}.csv".format(dataset, loss_name, seed)
    )
    for i in range(len(all_err)):
        decision_making_dic = {}
        decision_making_dic["y0"] = all_y0[i].item()
        decision_making_dic["decision_loss"] = all_loss[i].item()
        decision_making_dic["err"] = all_err[i].item()
        decision_making_dic["c"] = all_c[i].item()
        write_result(decision_making_results_file, decision_making_dic)


def report_recalibration_results(
    model,
    dataset,
    train_frac,
    loss_name,
    seed,
    posthoc_recalibration,
    recalibration_parameters,
    save,
    save_dir="results",
):
    results_file = save_dir + "/" + save + ".csv"
    result = {
        "dataset": dataset,
        "rmse": getattr(model, "rmse", 0),
        "loss": loss_name,
        "ece": getattr(model, "ece", 0),
        #              "stddev": getattr(model, "sharpness", 0),
        #              "point_unbiasedness_max": getattr(model, "point_unbiasedness_max", 0),
        #              "point_unbiasedness_mean": getattr(model, "point_unbiasedness_mean", 0),
        "threshold_calibration_error_less": getattr(
            model, "threshold_calibration_error_less", 0
        ),
        "threshold_calibration_error_greater": getattr(
            model, "threshold_calibration_error_greater", 0
        ),
        "threshold_calibration_error_both": getattr(
            model, "threshold_calibration_error_both", 0
        ),
        "threshold_calibration_error_all": getattr(
            model, "threshold_calibration_error_all", 0
        ),
        "point_calibration_error": getattr(model, "point_calibration_error", 0),
        "distribution_calibration_error": getattr(
            model, "distribution_calibration_error", 0
        ),
        "point_calibration_error_uniform_mass": getattr(
            model, "point_calibration_error_uniform_mass", 0
        ),
        "val_point_calibration_error": getattr(model, "val_point_calibration_error", 0),
        "val_point_calibration_error_uniform_mass": getattr(
            model, "val_point_calibration_error_uniform_mass", 0
        ),
        "train_point_calibration_error": getattr(
            model, "train_point_calibration_error", 0
        ),
        "train_point_calibration_error_uniform_mass": getattr(
            model, "train_point_calibration_error_uniform_mass", 0
        ),
        "train_true_vs_pred_loss": getattr(model, "train_true_vs_pred_loss", 0),
        #              "false_positive_rate_error": getattr(model, "false_positive_rate_error", 0),
        #              "false_negative_rate_error": getattr(model, "false_negative_rate_error", 0),
        "true_vs_pred_loss": getattr(model, "true_vs_pred_loss", 0),
        "decision_loss": getattr(model, "decision_loss", 0),
        "posthoc_recalibration": posthoc_recalibration,
        "train_frac": train_frac,
        "seed": seed,
    }
    for x in ["num_layers", "n_dim", "epochs", "n_bins", "flow_type"]:
        if recalibration_parameters and x in recalibration_parameters:
            result[x] = recalibration_parameters[x]
        else:
            result[x] = None
    write_result(results_file, result)

    all_err = getattr(model, "all_err", [])
    all_loss = getattr(model, "all_loss", [])
    all_y0 = getattr(model, "all_y0", [])
    all_c = getattr(model, "all_c", [])

    if recalibration_parameters == None:
        recalibration_parameters = {}

    decision_making_results_file = (
        save_dir
        + "/"
        + save
        + "_decision_{}_{}_{}_{}_{}_{}.csv".format(
            dataset,
            loss_name,
            posthoc_recalibration,
            recalibration_parameters.get("n_bins", 1),
            recalibration_parameters.get("num_layers", 0),
            seed,
        )
    )

    for i in range(len(all_err)):
        decision_making_dic = {}
        decision_making_dic["y0"] = all_y0[i].item()
        decision_making_dic["decision_loss"] = all_loss[i].item()
        decision_making_dic["err"] = all_err[i].item()
        decision_making_dic["c"] = all_c[i].item()
        write_result(decision_making_results_file, decision_making_dic)
