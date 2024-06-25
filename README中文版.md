# Stockformer 数据文件解释和执行方法

[English Version](README.md)

## “Stockformer”代码概览
本文题为“Stockformer：基于小波变换和多任务自注意力网络的价格-量因子股票选择模型”，目前正在《Expert Systems with Applications》期刊审稿中。您可以在SSRN阅读预印本原文：[https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4648073](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4648073)。

通过此链接，读者可以访问提交的完整论文文本，从而在出版前了解详细的方法论和研究发现。


## 原始数据集和划分后的子数据集
由于原始数据（包含360个量价因子）和处理后数据（包含360个量价因子）较大，作者已将其存储在云盘中，供读者使用，其中，`Stock_CN_2018-03-01_2024-03-01`为原始数据，其余文件夹为处理后子数据集。

原始数据链接： [raw_data](https://pan.baidu.com/s/1dnmzt9F2Ug9bCQDZwZ2e4Q?pwd=ykqp)

处理后总共有14个子数据集，为不同的时间段的回测提供相应的数据支持，详细的数据内容如下所示：

|             | 训练集     | 训练集     | 验证集     | 验证集     | 测试集     | 测试集     |
| ----------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 数据集      | 开始日期   | 结束日期   | 开始日期   | 结束日期   | 开始日期   | 结束日期   |
| 子数据集 1  | 2018-03-01 | 2020-02-28 | 2020-03-02 | 2020-06-30 | 2020-07-01 | 2020-10-29 |
| 子数据集 2  | 2018-05-31 | 2020-05-29 | 2020-06-01 | 2020-09-23 | 2020-09-24 | 2021-01-25 |
| 子数据集 3  | 2018-08-27 | 2020-08-26 | 2020-08-27 | 2020-12-25 | 2020-12-28 | 2021-04-28 |
| 子数据集 4  | 2018-11-28 | 2020-11-27 | 2020-11-30 | 2021-03-30 | 2021-03-31 | 2021-07-28 |
| 子数据集 5  | 2019-03-04 | 2021-03-02 | 2021-03-03 | 2021-06-30 | 2021-07-01 | 2021-11-01 |
| 子数据集 6  | 2019-06-03 | 2021-06-01 | 2021-06-02 | 2021-09-27 | 2021-09-28 | 2022-01-26 |
| 子数据集 7  | 2019-08-28 | 2021-08-26 | 2021-08-27 | 2021-12-28 | 2021-12-29 | 2022-05-05 |
| 子数据集 8  | 2019-11-29 | 2021-11-30 | 2021-12-01 | 2022-03-31 | 2022-04-01 | 2022-08-01 |
| 子数据集 9  | 2020-03-04 | 2022-03-03 | 2022-03-04 | 2022-07-04 | 2022-07-05 | 2022-11-02 |
| 子数据集 10 | 2020-06-03 | 2022-06-06 | 2022-06-07 | 2022-09-28 | 2022-09-29 | 2023-02-03 |
| 子数据集 11 | 2020-08-31 | 2022-08-30 | 2022-08-31 | 2022-12-29 | 2022-12-30 | 2023-05-05 |
| 子数据集 12 | 2020-12-02 | 2022-12-01 | 2022-12-02 | 2023-04-03 | 2023-04-04 | 2023-08-02 |
| 子数据集 13 | 2021-03-05 | 2023-03-06 | 2023-03-07 | 2023-07-05 | 2023-07-06 | 2023-11-03 |
| 子数据集 14 | 2021-06-04 | 2023-06-05 | 2023-06-06 | 2023-09-28 | 2023-10-09 | 2024-01-30 |


## 文件描述
- `data_processing_script`：原始数据的数据清洗。上传的数据已完成所有内容的生成。

  - `stockformer_input_data_processing`：生成Stockformer输入数据的数据处理脚本。
    - `data_Interception.py`：将原始数据集切分为不同的子集。
    - `Stockformer_data_preprocessing_script.py`：从子集内容生成模型所需的输入。
    - `results_data_processing.py`：将模型生成的结果处理成后续回测和结果展示所需的输出。

  - `volume_and_price_factor_construction`：量价因子的构建。
    - `1_stock_data_consolidation.ipynb`：整合原始数据。
    - `2_data_preprocessing.ipynb`：预处理整合后的数据。
    - `3_qlib_factor_construction.ipynb`：构建因子。
    - `4_neutralization.ipynb`：因子中性化。
    - `5_factor_verification.ipynb`：验证因子。

- `Stockformermodel`

  - Stockformer的神经网络架构：`Multitask_Stockformer_models.py`包括模型的各个组件，以及双任务输出的结果构造。

- `data`：请将下载下来的数据保存到本文件夹下。

  - `data`下的文件夹，每个都应该命名为：`Stock_CN_xxxx-xx-xx_xxxx-xx-xx`
  - `Stock_CN_xxxx-xx-xx_xxxx-xx-xx`
    - `label_processed.csv`：各个股票收益率数据集。
    - `Alpha_360_xxx-xx-xx_xxxx-xx-xx`：360个量价因子。
    - `corr_adj.npy`：输入数据的相关性矩阵（用于通过Struc2vec生成高维向量表达，具体生成方法参见[struc2vec](https://github.com/shenweichen/GraphEmbedding/blob/master/examples/struc2vec_flight.py)）。
    - `128_corr_struc2vec_adjgat.npy`：由Struc2vec生成的高维向量，可直接用于网络输入。
    - `flow.npz`：处理后的实际输入收益率数据。
    - `trend_indicator.npz`：处理后的实际输入涨跌趋势指标。

- `log/STOCK`
  - `log_Multitask_2020-12-02_2023-08-02`：神经网络输出的日志文件，现在保存了`2020-12-02至2023-08-02`的日志输出。

- `lib`

  - `Multitask_Stockformer_utils.py`：Stockformer输入数据的在训练时的处理和评估指标的建立。

- `output`：输出结果文件的文件夹，现在保存了`2020-12-02至2023-08-02`的输出数据。

- `config`：网络的配置文件。

- `cpt/STOCK`

  - `saved_model_Multitask_2020-12-02_2023-08-02`：保存的训练过的神经网络模型，现在保存了`2020-12-02至2023-08-02`的训练模型。

- `runs/Multitask_Stockformer`：该文件夹包含 TensorBoard 文件，其中包括训练的损失函数等训练的信息，现在保存了`2020-12-02至2023-08-02`的训练信息。

  - 通过运行：

    ```sh
    tensorboard --logdir='runs/Multitask_Stockformer/Stock_CN_2020-12-02_2023-08-02'
    ```


​				可查看相关信息。

- `Stockformer_train.py`：模型训练文件。
- `backtest`
  - `Backtest.ipynb`：基于[Qlib](https://github.com/microsoft/qlib)包，用于Stockformer以及对比的先进回测模型的topk-dropout回测代码。
- `.vscode`： 该文件夹包含 Visual Studio Code (VSCode) 集成开发环境 (IDE) 的配置文件。在 VSCode 中进行调试时，这些文件尤其有用。请注意，您可能需要修改这些配置中的文件路径，以便与您的自定义项目路径相匹配。


## 如何运行
在终端中执行以下命令以运行模型：

```sh
python Multitask_Stockformer_models.py --config Multitask_Stock.conf
```

## 引用

如果您在研究中使用了此模型或数据集，请按以下方式引用我们的论文：

Ma, Bohan; Xue, Yushan; Lu, Yuan; Chen, Jing. "Stockformer: A Price-Volume Factor Stock Selection Model Based on Wavelet Transform and Multi-Task Self-Attention Networks," June 17, 2024. Available at SSRN: [https://ssrn.com/abstract=4648073](https://ssrn.com/abstract=4648073) or DOI: [10.2139/ssrn.4648073](http://dx.doi.org/10.2139/ssrn.4648073)

此引用提供了所有必要的细节，如完整的作者名单、论文标题、发布日期和直接链接到论文的链接，便于访问和验证。
