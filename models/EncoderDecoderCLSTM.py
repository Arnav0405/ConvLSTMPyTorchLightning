import torch.nn as nn
import torch

from models.ConvLSTMCell import ConvLSTMCell

class EncoderDecoderCLSTM(nn.Module):
    def __init__(self, nf, in_chan, num_classes=10, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan, hidden_dim=nf, kernel_size=(3, 3), bias=True)
        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True)
        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.classifier = nn.Sequential(
            nn.Linear(nf, nf // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(nf // 2, num_classes),
            nn.Softmax()
        )

    def autoEncoder(self, x, seq_len, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):
        # Encoder at time step i:
        for i in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, i, :, :, :], cur_state=[h_t, c_t])  
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t, cur_state=[h_t2, c_t2])   

        encoder_vector = h_t2

        # Decoder at future time step i:
        for i in range(seq_len // 10):   # Run decoder for half the input sequence length
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector, cur_state=[h_t3, c_t3])  
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3, cur_state=[h_t4, c_t4])  
            dencoder_vector = h_t4
        
        output = self.global_pool(dencoder_vector)
        classifier_output = output.view(output.size(0), -1)  # Flatten

        classifier_outputs = self.classifier(classifier_output)

        return classifier_outputs

    def forward(self, x, **kwargs):
        # input_tensor: 5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width

        b, seq_length, _, h, w = x.size()
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        outputs = self.autoEncoder(x, seq_length, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs