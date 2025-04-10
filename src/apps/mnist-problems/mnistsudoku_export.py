import logging



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')




def exportResultsSudoku(samples, avg_inference_runtime, probposs_transformation_method, n_train, n_test, size):
    
    exclude_keys = ['avg_inference_runtime', 'probposs_transformation_method', 'id', 'trainsize',
                    "roc_auc_dict_train", "roc_auc_dict_test"]
   
    all_keys = set(samples[0].keys()) - set(exclude_keys)
   
    data_values = {key: [] for key in all_keys}
    
    for sample in samples:
        for key in all_keys:
            data_values[key].append(sample[key])

    
    statistics = {}
    for key, values in data_values.items():
        statistics[key] = {
            'average': np.average(values),
            'std_dev': np.std(values, ddof=1)
        }

    statistics["average_inference_runtime"] = {
        'average': np.average(avg_inference_runtime),
        'std_dev': np.std(avg_inference_runtime, ddof=1)
    }

    
    logging.info("Probability-possibility transformation:" + str(probposs_transformation_method))
    logging.info("n_train:" + str(n_train))
    logging.info("n_test:" + str(n_test))

    data_export = {}
    data_export["n_train"] = n_train
    data_export["n_test"] = n_test
    data_export["probposs_transformation_method"] = probposs_transformation_method
    if len(pair) > 0:
        data_export["pair"] = pair
    for key, stats in statistics.items():
        logging.info(f"{key}: Average = {stats['average']}, Standard Deviation = {stats['std_dev']}")
        data_export[str(key) + "_average"] = stats['average']
        data_export[str(key) + "_std"] = stats["std_dev"]
    logging.info(data_export)

    id_pinesy = 1 if probposs_transformation_method == 1 else 2
    filepath = (BASE_FILE_PATH_RESULTS + str(id_pinesy) + "/experiment::mnist-"+str(size)+"x"+str(size)+"/" + "/-result-"+ str(datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")) + str(".csv"))
    directory, filename = os.path.split(filepath)
    os.makedirs(directory, exist_ok=True)

    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_export.keys())
        writer.writeheader()
        writer.writerow(data_export)
