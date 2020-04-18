import numpy as np
import sys
import csv
from torch.utils.data import Dataset
csv.fileld_size_limit(sys.maxsize)

class MyDataset(Dataset):
    def __init__(self, data_path, max_length=1014):
        self.data_path = data_path
        self.vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        # 単位行列の作成
        self.identity_mat = np.identity(len(self.vocabulary))
        # csvファイルの読み込み
        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx
                    text += " "
                # labelを0始まりにする
                label = int(line[0]) - 1
                # textとlabelをリストに追加
                texts.append(text)
                labels.append(label)
        # textとlabelを格納
        self.texts = texts
        self.labels = labels
        # textの最大長を格納
        self.max_length = max_length
        # データの長さを格納
        self.length = len(self.labels)
        # クラス数を格納（重複する要素を取り除きクラス数を計算）
        self.num_classes = len(set(self.labels))

    def __len__(self):
        # データ長を返却
        return self.length

    def __getitem__(self, index):
        # indexを指定してtextの中身を取り出す
        raw_text = self.texts[index]
        # raw_textの各文字のindexに1を立てたベクトル（one hot vector）を取得
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary], dtype=np.float32)
        # データ長が最大長よりも長い場合は切り捨て
        if len(data) > self.max_length:
            data = data[:self.max_length]
        # データ長が0より大きく最大長未満の場合は残りを0でpadding
        elif 0 < len(data) < self.max_length:
            data = np.concatenate((data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        # データ長が0の場合は全て0でpadding
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
        # 対応するラベルを取得
        label = self.labels[index]
        return data, label
