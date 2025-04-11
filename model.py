import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height, num_classes):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.expected_height = img_height // 8
        self.rnn_input_size = 256
        self.rnn_hidden_size = 128

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(self.rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        assert h == self.expected_height, f"Beklenmeyen yükseklik! H = {h}, beklenen = {self.expected_height}"
        x = x.permute(0, 3, 1, 2)
        x = x.view(b, w, c)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
