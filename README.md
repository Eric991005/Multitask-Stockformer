# Stockformer Data File Explanation and Execution Methods

[中文版本](README中文版.md)

## "Stockformer" Code Overview
This paper, titled "Stockformer: A Price-Volume Factor Stock Selection Model Based on Wavelet Transform and Multi-Task Self-Attention Networks," is currently under review at Expert Systems with Applications. You can read the preprint version of the paper on SSRN: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4648073](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4648073).

This link provides access to the full text of the paper as submitted, allowing readers to explore the detailed methodology and findings prior to publication.


## Original Dataset and Sub-Datasets After Partition
Due to the large size of both the original data (which contains 360 price and volume factors) and the processed data (which also contains 360 factors), the author has stored them on a cloud drive for readers to use. The folder `Stock_CN_2018-03-01_2024-03-01` contains the original data, while other folders hold the processed sub-datasets.

Original data link: [raw_data](https://pan.baidu.com/s/1dnmzt9F2Ug9bCQDZwZ2e4Q?pwd=ykqp)

There are a total of 14 sub-datasets after processing, providing data support for backtesting over different time periods. The detailed contents of these datasets are as follows:

|             | Training Set | Training Set | Validation Set | Validation Set | Test Set    | Test Set    |
| ----------- | ------------ | ------------ | -------------- | -------------- | ----------- | ----------- |
| Dataset     | Start Date   | End Date     | Start Date     | End Date       | Start Date  | End Date    |
| Subset 1    | 2018-03-01   | 2020-02-28   | 2020-03-02     | 2020-06-30     | 2020-07-01  | 2020-10-29  |
| Subset 2    | 2018-05-31   | 2020-05-29   | 2020-06-01     | 2020-09-23     | 2020-09-24  | 2021-01-25  |
| Subset 3    | 2018-08-27   | 2020-08-26   | 2020-08-27     | 2020-12-25     | 2020-12-28  | 2021-04-28  |
| Subset 4    | 2018-11-28   | 2020-11-27   | 2020-11-30     | 2021-03-30     | 2021-03-31  | 2021-07-28  |
| Subset 5    | 2019-03-04   | 2021-03-02   | 2021-03-03     | 2021-06-30     | 2021-07-01  | 2021-11-01  |
| Subset 6    | 2019-06-03   | 2021-06-01   | 2021-06-02     | 2021-09-27     | 2021-09-28  | 2022-01-26  |
| Subset 7    | 2019-08-28   | 2021-08-26   | 2021-08-27     | 2021-12-28     | 2021-12-29  | 2022-05-05  |
| Subset 8    | 2019-11-29   | 2021-11-30   | 2021-12-01     | 2022-03-31     | 2022-04-01  | 2022-08-01  |
| Subset 9    | 2020-03-04   | 2022-03-03   | 2022-03-04     | 2022-07-04     | 2022-07-05  | 2022-11-02  |
| Subset 10   | 2020-06-03   | 2022-06-06   | 2022-06-07     | 2022-09-28     | 2022-09-29  | 2023-02-03  |
| Subset 11   | 2020-08-31   | 2022-08-30   | 2022-08-31     | 2022-12-29     | 2022-12-30  | 2023-05-05  |
| Subset 12   | 2020-12-02   | 2022-12-01   | 2022-12-02     | 2023-04-03     | 2023-04-04  | 2023-08-02  |
| Subset 13   | 2021-03-05   | 2023-03-06   | 2023-03-07     | 2023-07-05     | 2023-07-06  | 2023-11-03  |
| Subset 14   | 2021-06-04   | 2023-06-05   | 2023-06-06     | 2023-09-28     | 2023-10-09  | 2024-01-30  |

## File Description
- `data_processing_script`: Data cleaning of the original data. The uploaded data has already completed all content generation.

  - `data_Interception.py`: Splits the original dataset into various subsets.
  - `Stockformer_data_preprocessing_script.py`: Generates the inputs needed for the model from the contents of the subsets.
  - `results_data_processing.py`: Processes the results generated by the model into outputs needed for subsequent backtesting and result display.

- `Stockformermodel`

  - Neural network architecture of Stockformer: `Multitask_Stockformer_models.py` includes various components of the model and the construction of dual-task output results.

- `data`: Please save the downloaded data to this folder.

  - Each folder under `data` should be named: `Stock_CN_xxxx-xx-xx_xxxx-xx-xx`
  - `Stock_CN_xxxx-xx-xx_xxxx-xx-xx`
    - `label_processed.csv`: Yield rate datasets for various stocks.
    - `Alpha_360_xxx-xx-xx_xxxx-xx-xx`: 360 volume-price factors.
    - `corr_adj.npy`: Correlation matrix of the input data (used to generate high-dimensional vector representations via Struc2vec, see [struc2vec](https://github.com/shenweichen/GraphEmbedding/blob/master/examples/struc2vec_flight.py) for specific generation methods).
    - `128_corr_struc2vec_adjgat.npy`: High-dimensional vectors generated by Struc2vec, can be directly used as network input.
    - `flow.npz`: Processed actual input yield rate data.
    - `trend_indicator.npz`: Processed actual input trend indicators.

- `log/STOCK`
  - `log_Multitask_2020-12-02_2023-08-02`: Log files output by the neural network, currently saving log outputs from `2020-12-02 to 2023-08-02`.

- `lib`

  - `Multitask_Stockformer_utils.py`: Processing of Stockformer input data during training and establishment of evaluation metrics.

- `output`: Folder for output result files, currently saving output data from `2020-12-02 to 2023-08-02`.

- `config`: Configuration files for the network.

- `cpt/STOCK`

  - `saved_model_Multitask_2020-12-02_2023-08-02`: Saved trained neural network models, currently saving trained models from `2020-12-02 to 2023-08-02`.

- `runs/Multitask_Stockformer`: This folder contains TensorBoard files, including training information such as the loss function, currently saving training information from `2020-12-02 to 2023-08-02`.

  - By running:

    ```sh
    tensorboard --logdir='runs/Multitask_Stockformer/Stock_CN_2020-12-02_2023-08-02'
    ```

    You can view related information.

- `Stockformer_train.py`: Model training file.
- `backtest`
  - `Backtest.ipynb`: Topk-dropout backtest code based on the [Qlib](https://github.com/microsoft/qlib) package, used for backtesting of Stockformer and advanced comparison models.
- `.vscode`: This folder contains configuration files for the Visual Studio Code (VSCode) Integrated Development Environment (IDE). These files are especially useful when debugging in VSCode. Please note that you may need to modify the file paths in these configurations to match your custom project paths.

## How to Run
Execute the following command in the terminal to run the model:

```sh
python Multitask_Stockformer_models.py --config Multitask_Stock.conf
```

## Citation

If you use this model or the dataset in your research, please cite our paper as follows:

Ma, Bohan; Xue, Yushan; Lu, Yuan; Chen, Jing. "Stockformer: A Price-Volume Factor Stock Selection Model Based on Wavelet Transform and Multi-Task Self-Attention Networks," June 17, 2024. Available at SSRN: [https://ssrn.com/abstract=4648073](https://ssrn.com/abstract=4648073) or DOI: [10.2139/ssrn.4648073](http://dx.doi.org/10.2139/ssrn.4648073)

This citation provides all the necessary details such as the full list of authors, the title of the paper, the publication date, and direct links to the paper for easy access and verification.
