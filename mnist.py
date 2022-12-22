import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from seed import set_seed


def get_grad(feature):
    # feature = feature[:-1]
    grad = feature[-1].grad
    feature = feature[:-1]
    return grad, feature
    
    
set_seed(0)
#----------------------------------------------------------
# ハイパーパラータなどの設定値
num_epochs = 10         # 学習を繰り返す回数
num_batch = 100         # 一度に処理する画像の枚数
learning_rate = 0.001   # 学習率
image_size = 28*28      # 画像の画素数(幅x高さ)

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------
# 学習用／評価用のデータセットの作成

# 変換方法の指定
transform = transforms.Compose([
    transforms.ToTensor()
    ])

# MNISTデータの取得
# https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
# 学習用
train_dataset = datasets.MNIST(
    './data',               # データの保存先
    train = True,           # 学習用データを取得する
    download = True,        # データが無い時にダウンロードする
    transform = transform   # テンソルへの変換など
    )
# 評価用
test_dataset = datasets.MNIST(
    './data', 
    train = False,
    transform = transform
    )

# データローダー
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = num_batch,
    shuffle = True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,     
    batch_size = num_batch,
    shuffle = True)

#----------------------------------------------------------
# ニューラルネットワークモデルの定義
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, 50, bias=False)
        self.fc2 = nn.Linear(50, 120, bias=False)
        self.fc3 = nn.Linear(120, 50, bias=False)
        self.fc4 = nn.Linear(50, output_size, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.net = nn.Sequential(
            nn.Sequential(self.fc1, nn.ReLU()),
            nn.Sequential(self.fc2, nn.ReLU()),
            nn.Sequential(self.fc3, nn.ReLU()),
            nn.Sequential(self.fc4, self.softmax),
        )

    def forward(self, x, layer_num=None):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        if layer_num != None:
            x = self.net[layer_num](x)
        else:
            x = self.net(x)          
        return x

#----------------------------------------------------------
# ニューラルネットワークの生成
model = Net(image_size, 10).to(device)

#----------------------------------------------------------
# 損失関数の設定
criterion = nn.CrossEntropyLoss() 

#----------------------------------------------------------
# 最適化手法の設定
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

#----------------------------------------------------------
# 学習
model.train()  # モデルを訓練モードにする

for epoch in range(num_epochs): # 学習を繰り返し行う
    loss_sum = 0

    for inputs, labels in train_dataloader:
        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        ft = []
        ft.append(inputs.cpu())

        for i in range(len(model.net)):
            for j in range(len(model.net) - 1):
                model.net[j].requires_grad = False
            model.net[-1].requires_grad = True
            ft[-1] = Variable(ft[-1], requires_grad=True)
            ft.append(model(ft[-1].to(device), i).cpu())

        loss = criterion(ft[-1].to(device), labels)
        loss_sum += loss
        loss.backward()

        ft = ft[:-1]
        grad, ft = get_grad(ft)
        optimizer.step()
        model.net[-1].requires_grad= False

        for i in reversed(range(len(model.net)-1)):
            optimizer.zero_grad()
            for j in range(i):
                model.net[0].requires_grad = False
            model.net[i].requires_grad = True
            output = model(ft[-1].to(device), i)
            loss = torch.sum((grad.to(device) * output), dim=1).mean()
            loss.backward()
            assert grad.shape[1] == output.shape[1], 'difference size error'
            grad, ft = get_grad(ft)
            optimizer.step()
            model.net[i].requires_grad = False

    # 学習状況の表示
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")

    # モデルの重みの保存
    torch.save(model.state_dict(), 'model_weights.pth')

#----------------------------------------------------------
# 評価
model.eval()  # モデルを評価モードにする

loss_sum = 0
correct = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss_sum += criterion(outputs, labels)

        # 正解の値を取得
        pred = outputs.argmax(1)
        # 正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")