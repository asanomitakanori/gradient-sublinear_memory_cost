
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, 120, bias=False)
        self.fc2 = nn.Linear(120, output_size, bias=False)
        self.fc3 = nn.Linear(output_size, output_size, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.net = nn.Sequential(
            nn.Sequential(self.fc1, nn.ReLU()),
            nn.Sequential(nn.ReLU(), self.fc2),
            nn.Sequential(self.fc3, self.softmax),
        )

    def forward(self, x, layer_num=None):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        if layer_num != None:
            x = self.net[layer_num](x)
        else:
            x = self.net(x)          
        return x