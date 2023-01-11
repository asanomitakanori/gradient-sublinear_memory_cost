import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from seed import set_seed
from model import Net


def get_grad(feature):
    # feature = feature[:-1]
    grad = feature[-1].grad
    feature = feature[:-1]
    return grad, feature
    

set_seed(0)
num_epochs = 10         # 学習を繰り返す回数
num_batch = 100         # 一度に処理する画像の枚数
learning_rate = 0.001   # 学習率
image_size = 28*28      # 画像の画素数(幅x高さ)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


model = Net(image_size, 10).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 
model.train()  # モデルを訓練モードにする


for epoch in range(num_epochs): # 学習を繰り返し行う
    loss_sum = 0

    for inputs, labels in train_dataloader:
        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える

        #----------------------------------------------------------
        ft = []
        grad = []
        ft.append(inputs)

        for i in range(len(model.net)):
            for j in range(len(model.net)):
                model.net[j].requires_grad = False
            ft[-1] = Variable(ft[-1], requires_grad=True)
            ft.append(model(ft[-1], i))

        loss = criterion(ft[-1], labels)
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
            output = output.detach().cpu()
        #----------------------------------------------------------

    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")
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