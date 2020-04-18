import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import shutil

from src.utils import *
from src.dataset import MyDataset
from src.character_level_cnn import CharacterLevelCNN


def get_args():
    # 何を実装したかの説明
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Character-level convolutional networks for text classification""")
    # データに含まれる文字列の種類
    parser.add_argument("-a", "--alphabet", type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    # データの長さの最大値
    parser.add_argument("-m", "--max_length", type=int, default=1014)
    # データの大小指定
    parser.add_argument("-f", "--feature", type=str, choices=["large", "small"], default="small",
                        help="small for 256 conv feature map, large for 1024 conv feature map")
    # 最適化方法の指定
    parser.add_argument("-p", "--optimizer", type=str, choices=["sgd", "adam"], default="sgd")
    # バッチサイズの指定
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    # エポックの数
    parser.add_argument("-n", "--num_epochs", type=int, default=20)
    # 学習係数
    parser.add_argument("-l", "--lr", type=float, default=0.001)  # recommended learning rate for sgd is 0.01, while for adam is 0.001
    # 学習させるデータセットの指定
    parser.add_argument("-d", "--dataset", type=str,
                        choices=["agnews", "dbpedia", "yelp_review", "yelp_review_polarity", "amazon_review",
                                 "amazon_polarity", "sogou_news", "yahoo_answers"], default="yelp_review_polarity",
                        help="public dataset used for experiment. If this parameter is set, parameters input and output are ignored")
    # early stoppingさせるためのロスの閾値
    parser.add_argument("-y", "--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    # early stoppingさせるためのエポック数の指定（nエポック繰り返してロスが下がらなかったら止める）
    parser.add_argument("-w", "--es_patience", type=int, default=3,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    # 入力するデータのパスを指定
    parser.add_argument("-i", "--input", type=str, default="input", help="path to input folder")
    # 学習結果の出力先のパスを指定
    parser.add_argument("-o", "--output", type=str, default="output", help="path to output folder")
    # tensorboardの記録先のパスを指定
    parser.add_argument("-v", "--log_path", type=str, default="tensorboard/char-cnn")
    # 引数を解析
    args = parser.parse_args()
    return args


def train(opt):
    # GPUが使えるか確認
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # データセットのパスを指定
    if opt.dataset in ["agnews", "dbpedia", "yelp_review", "yelp_review_polarity", "amazon_review",
                       "amazon_polarity", "sogou_news", "yahoo_answers"]:
        opt.input, opt.output = get_default_folder(opt.dataset, opt.feature)
    # outputのディレクトリが存在していなかったらディレクトリを作成
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    # あとで消す
    print(os.sep)
    print(opt.input)
    # outputファイルを作成
    output_file = open(opt.output + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    # パラメータの設定
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "num_workers": 0}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "num_workers": 0}
    # データセットの読み込み
    training_set = MyDataset(opt.input + os.sep + "train.csv", opt.max_length)
    test_set = MyDataset(opt.input + os.sep + "test.csv", opt.max_length)
    training_generator = DataLoader(training_set, **training_params)
    test_generator = DataLoader(test_set, **test_params)
    # データセットの大小指定
    if opt.feature == "small":
        model = CharacterLevelCNN(input_length=opt.max_length, n_classes=training_set.num_classes,
                                  input_dim=len(opt.alphabet),
                                  n_conv_filters=256, n_fc_neurons=1024)

    elif opt.feature == "large":
        model = CharacterLevelCNN(input_length=opt.max_length, n_classes=training_set.num_classes,
                                  input_dim=len(opt.alphabet),
                                  n_conv_filters=1024, n_fc_neurons=2048)
    else:
        sys.exit("Invalid feature mode!")
    # ログのパス指定とログの書き込み
    log_path = "{}_{}_{}".format(opt.log_path, opt.feature, opt.dataset)
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    # GPUが使えるか確認
    if torch.cuda.is_available():
        model.cuda()
    # 損失関数にはクロスエントロピーを使用
    criterion = nn.CrossEntropyLoss()
    # 最適化方法の指定
    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    # パラメータの指定
    best_loss = 1e5
    best_epoch = 0
    # 学習
    model.train()
    # エポックごとのイテレーション回数
    num_iter_per_epoch = len(training_generator)
    # 学習開始
    for epoch in range(opt.num_epochs):
        for iter, batch in enumerate(training_generator):
            # featureとlabelに分ける（train）
            feature, label = batch
            # GPU使用可能か確認
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            # 勾配の初期化
            optimizer.zero_grad()
            # 予測
            predictions = model(feature)
            # ロスの計算
            loss = criterion(predictions, label)
            # 勾配の計算
            loss.backward()
            # パラメータの更新
            optimizer.step()
            # モデルの評価
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                              list_metrics=["accuracy"])
            # 現在の学習状況の表示
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epochs,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        # 推論モードへの切り替え
        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for batch in test_generator:
            # featureとlabelに分ける（test）
            te_feature, te_label = batch
            # サンプル数の取得
            num_sample = len(te_label)
            # GPUが使用可能か確認
            if torch.cuda.is_available():
                te_feature = te_feature.cuda()
                te_label = te_label.cuda()
            # パラメータの保存を止める
            with torch.no_grad():
                te_predictions = model(te_feature)
            # ロスの計算
            te_loss = criterion(te_predictions, te_label)
            # 現在のロスをリストに追加
            loss_ls.append(te_loss * num_sample)
            # テストデータのlabelをリストに追加
            te_label_ls.extend(te_label.clone().cpu())
            # テストデータの予測値をリストに追加
            te_pred_ls.append(te_predictions.clone().cpu())
        # バッチ全体でのロスを計算
        te_loss = sum(loss_ls) / test_set.__len__()
        # testの予測値のtensorを縦方向に連結
        te_pred = torch.cat(te_pred_ls, 0)
        # testのlabelをnumpy.array化
        te_label = np.array(te_label_ls)
        # モデルの評価
        test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
        # 出力結果をファイルに書き込み
        output_file.write(
            "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                epoch + 1, opt.num_epochs,
                te_loss,
                test_metrics["accuracy"],
                test_metrics["confusion_matrix"]))
        # 現在のテストの評価状況を表示
        print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            opt.num_epochs,
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["accuracy"]))
        writer.add_scalar('Test/Loss', te_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
        # 学習モードに変更
        model.train()
        # 現在のロスがあらかじめ設定したロスの閾値を下回ったら，現在のモデルを保存
        if te_loss + opt.es_min_delta < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            torch.save(model, "{}/char-cnn_{}_{}".format(opt.output, opt.dataset, opt.feature))
        # Early stopping
        if epoch - best_epoch > opt.es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch, te_loss, best_epoch))
            break
        # 勾配グリッピング
        if opt.optimizer == "sgd" and epoch % 3 == 0 and epoch > 0:
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            current_lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr


if __name__ == "__main__":
    opt = get_args()
    train(opt)
