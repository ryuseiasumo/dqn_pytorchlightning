from torch import nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(
        self,
        input_size: int = 4, #状態数 or 観測数
        lin1_size: int = 128,
        lin2_size: int = 128,
        output_size:int = 2, #各状態における行動候補数
    ):
        super().__init__()
        
        self.l1 = nn.Linear(input_size, lin1_size)
        self.l2 = nn.Linear(lin1_size, lin2_size)
        self.l3 = nn.Linear(lin2_size, output_size) #行動候補数が各入力に対する出力サイズ → 入力した状態における各行動でのQ関数の値を出力する

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x