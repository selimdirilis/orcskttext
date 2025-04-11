import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_height, num_classes):
        super(CRNN, self).__init__()

        # CNN: Özellik çıkarımı
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # H/8
        )

        self.expected_height = img_height // 8  # bu sabit kalmalı
        self.rnn_input_size = 256
        self.rnn_hidden_size = 128

        # RNN
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Fully Connected
        self.fc = nn.Linear(self.rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()

        # 🔥 Buradaki kontrolü expected_height ile yap
        assert h == self.expected_height, f"Beklenmeyen yükseklik! H = {h}, beklenen = {self.expected_height}"

        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.view(b, w, c)  # (B, W, C)

        x, _ = self.rnn(x)  # (B, W, 2*hidden)
        x = self.fc(x)  # (B, W, num_classes)

        return x
