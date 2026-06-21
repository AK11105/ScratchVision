import torch.nn as nn 
from ..components.googlenet.Inception import Inception

class GoogLeNet(nn.Module):
    def __init__(
        self,
        input_dim=3,
        x7_output_dim=64,  # input to next depth 2 conv (1x1 + 3x3)
        x3_output_dim=192,  # input to first inception block
        # inception3a
        i1_x1=64,
        i1_x3=128,
        i1_x5=32,
        i1_dim3=96,
        i1_dim5=16,
        i1_pp=32,
        # inception3b (input will be sum of above = 256)
        i2_x1=128,
        i2_x3=192,
        i2_x5=96,
        i2_dim3=128,
        i2_dim5=32,
        i2_pp=64,
        # inception4a
        i3_x1=192,
        i3_x3=208,
        i3_x5=48,
        i3_dim3=96,
        i3_dim5=16,
        i3_pp=64,
        # inception4b
        i4_x1=160,
        i4_x3=224,
        i4_x5=64,
        i4_dim3=112,
        i4_dim5=24,
        i4_pp=64,
        # inception4c
        i5_x1=128,
        i5_x3=256,
        i5_x5=64,
        i5_dim3=128,
        i5_dim5=28,
        i5_pp=64,
        # inception4d
        i6_x1=112,
        i6_x3=288,
        i6_x5=64,
        i6_dim3=144,
        i6_dim5=32,
        i6_pp=64,
        # inception4e
        i7_x1=256,
        i7_x3=320,
        i7_x5=128,
        i7_dim3=160,
        i7_dim5=32,
        i7_pp=128,
        # inception5a
        i8_x1=256,
        i8_x3=320,
        i8_x5=128,
        i8_dim3=160,
        i8_dim5=32,
        i8_pp=128,
        # inception5b
        i9_x1=384,
        i9_x3=384,
        i9_x5=128,
        i9_dim3=192,
        i9_dim5=48,
        i9_pp=128,
        # task specific final output channel
        output_dim=10,
        # aux conv
        aux_op=128,
        # training signal
        training=False,
    ):
        super(GoogLeNet, self).__init__()

        self.input_dim = input_dim
        self.x7_output_dim = x7_output_dim
        self.x3_output_dim = x3_output_dim
        # inception3a
        self.i1_x1 = i1_x1
        self.i1_x3 = i1_x3
        self.i1_x5 = i1_x5
        self.i1_dim3 = i1_dim3
        self.i1_dim5 = i1_dim5
        self.i1_pp = i1_pp
        # inception3b
        self.i2_x1 = i2_x1
        self.i2_x3 = i2_x3
        self.i2_x5 = i2_x5
        self.i2_dim3 = i2_dim3
        self.i2_dim5 = i2_dim5
        self.i2_pp = i2_pp
        # inception4a
        self.i3_x1 = i3_x1
        self.i3_x3 = i3_x3
        self.i3_x5 = i3_x5
        self.i3_dim3 = i3_dim3
        self.i3_dim5 = i3_dim5
        self.i3_pp = i3_pp
        # inception4b
        self.i4_x1 = i4_x1
        self.i4_x3 = i4_x3
        self.i4_x5 = i4_x5
        self.i4_dim3 = i4_dim3
        self.i4_dim5 = i4_dim5
        self.i4_pp = i4_pp
        # inception4c
        self.i5_x1 = i5_x1
        self.i5_x3 = i5_x3
        self.i5_x5 = i5_x5
        self.i5_dim3 = i5_dim3
        self.i5_dim5 = i5_dim5
        self.i5_pp = i5_pp
        # inception4d
        self.i6_x1 = i6_x1
        self.i6_x3 = i6_x3
        self.i6_x5 = i6_x5
        self.i6_dim3 = i6_dim3
        self.i6_dim5 = i6_dim5
        self.i6_pp = i6_pp
        # inception4e
        self.i7_x1 = i7_x1
        self.i7_x3 = i7_x3
        self.i7_x5 = i7_x5
        self.i7_dim3 = i7_dim3
        self.i7_dim5 = i7_dim5
        self.i7_pp = i7_pp
        # inception5a
        self.i8_x1 = i8_x1
        self.i8_x3 = i8_x3
        self.i8_x5 = i8_x5
        self.i8_dim3 = i8_dim3
        self.i8_dim5 = i8_dim5
        self.i8_pp = i8_pp
        # inception5b
        self.i9_x1 = i9_x1
        self.i9_x3 = i9_x3
        self.i9_x5 = i9_x5
        self.i9_dim3 = i9_dim3
        self.i9_dim5 = i9_dim5
        self.i9_pp = i9_pp
        # output
        self.output_dim = output_dim
        # aux
        self.aux_op = aux_op

        # train signal
        self.training = training

        self.activation = nn.ReLU()
        self.mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fpool = nn.AvgPool2d(
            kernel_size=7, stride=1, padding=0
        )  # assumes final op before avg pool will be 7x7xc
        self.dropout = nn.Dropout(p=0.4)
        self.lrn = nn.LocalResponseNorm(k=2, alpha=1e-4, beta=0.75, size=5)
        self.auxpool = nn.AvgPool2d(kernel_size=5, stride=3, padding=0)
        self.flatten = nn.Flatten()

        # Layer Building
        # First conv
        self.convx7 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=x7_output_dim,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        # depth 2 conv
        self.convx1 = nn.Conv2d(
            in_channels=x7_output_dim,
            out_channels=x7_output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.convx3 = nn.Conv2d(
            in_channels=x7_output_dim,
            out_channels=x3_output_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # inception 3a
        self.inception3a = Inception(
            input_dim=x3_output_dim,
            x1=self.i1_x1,
            x3=self.i1_x3,
            x5=self.i1_x5,
            dim3=self.i1_dim3,
            dim5=self.i1_dim5,
            pp=self.i1_pp,
        )
        # inception 3b
        self.ip_3b = self.i1_x1 + self.i1_x3 + self.i1_x5 + self.i1_pp
        self.inception3b = Inception(
            input_dim=self.ip_3b,
            x1=self.i2_x1,
            x3=self.i2_x3,
            x5=self.i2_x5,
            dim3=self.i2_dim3,
            dim5=self.i2_dim5,
            pp=self.i2_pp,
        )
        # inception 4a
        self.ip_4a = self.i2_x1 + self.i2_x3 + self.i2_x5 + self.i2_pp
        self.inception4a = Inception(
            input_dim=self.ip_4a,
            x1=self.i3_x1,
            x3=self.i3_x3,
            x5=self.i3_x5,
            dim3=self.i3_dim3,
            dim5=self.i3_dim5,
            pp=self.i3_pp,
        )
        # inception 4b
        self.ip_4b = self.i3_x1 + self.i3_x3 + self.i3_x5 + self.i3_pp
        self.inception4b = Inception(
            input_dim=self.ip_4b,
            x1=self.i4_x1,
            x3=self.i4_x3,
            x5=self.i4_x5,
            dim3=self.i4_dim3,
            dim5=self.i4_dim5,
            pp=self.i4_pp,
        )
        # inception 4c
        self.ip_4c = self.i4_x1 + self.i4_x3 + self.i4_x5 + self.i4_pp
        self.inception4c = Inception(
            input_dim=self.ip_4c,
            x1=self.i5_x1,
            x3=self.i5_x3,
            x5=self.i5_x5,
            dim3=self.i5_dim3,
            dim5=self.i5_dim5,
            pp=self.i5_pp,
        )
        # inception 4d
        self.ip_4d = self.i5_x1 + self.i5_x3 + self.i5_x5 + self.i5_pp
        self.inception4d = Inception(
            input_dim=self.ip_4d,
            x1=self.i6_x1,
            x3=self.i6_x3,
            x5=self.i6_x5,
            dim3=self.i6_dim3,
            dim5=self.i6_dim5,
            pp=self.i6_pp,
        )
        # inception 4e
        self.ip_4e = self.i6_x1 + self.i6_x3 + self.i6_x5 + self.i6_pp
        self.inception4e = Inception(
            input_dim=self.ip_4e,
            x1=self.i7_x1,
            x3=self.i7_x3,
            x5=self.i7_x5,
            dim3=self.i7_dim3,
            dim5=self.i7_dim5,
            pp=self.i7_pp,
        )
        # inception 5a
        self.ip_5a = self.i7_x1 + self.i7_x3 + self.i7_x5 + self.i7_pp
        self.inception5a = Inception(
            input_dim=self.ip_5a,
            x1=self.i8_x1,
            x3=self.i8_x3,
            x5=self.i8_x5,
            dim3=self.i8_dim3,
            dim5=self.i8_dim5,
            pp=self.i8_pp,
        )
        # inception 5b
        self.ip_5b = self.i8_x1 + self.i8_x3 + self.i8_x5 + self.i8_pp
        self.inception5b = Inception(
            input_dim=self.ip_5b,
            x1=self.i9_x1,
            x3=self.i9_x3,
            x5=self.i9_x5,
            dim3=self.i9_dim3,
            dim5=self.i9_dim5,
            pp=self.i9_pp,
        )
        # final classifier
        self.ip_fc = self.i9_x1 + self.i9_x3 + self.i9_x5 + self.i9_pp
        self.fc = nn.Linear(in_features=self.ip_fc, out_features=output_dim)

        # auxillary classifiers
        # 4a classifier
        self.aux_4a_ip = self.i3_x1 + self.i3_x3 + self.i3_x5 + self.i3_pp
        self.aux_4a_convx1 = nn.Conv2d(
            in_channels=self.aux_4a_ip,
            out_channels=aux_op,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.aux_4a_fc1 = nn.Linear(
            in_features=aux_op * 4 * 4, out_features=1024
        )  # assumes final image size to 4x4x128, on flattening 1024
        self.aux_4a_fc2 = nn.Linear(in_features=1024, out_features=output_dim)
        self.aux_4a_dropout = nn.Dropout(p=0.7)
        # 4d classifier
        self.aux_4d_ip = self.i6_x1 + self.i6_x3 + self.i6_x5 + self.i6_pp
        self.aux_4d_convx1 = nn.Conv2d(
            in_channels=self.aux_4d_ip,
            out_channels=aux_op,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.aux_4d_fc1 = nn.Linear(
            in_features=aux_op * 4 * 4, out_features=1024
        )  # assumes final image size to 4x4x128, on flattening 1024
        self.aux_4d_fc2 = nn.Linear(in_features=1024, out_features=output_dim)
        self.aux_4d_dropout = nn.Dropout(p=0.7)

    def forward(self, X):
        X = self.convx7(X)
        X = self.activation(X)
        X = self.lrn(X)
        X = self.mpool(X)
        X = self.convx1(X)
        X = self.activation(X)
        X = self.convx3(X)
        X = self.activation(X)
        X = self.lrn(X)
        X = self.mpool(X)
        X = self.inception3a(X)
        X = self.inception3b(X)
        X = self.mpool(X)
        X = self.inception4a(X)

        # aux classifier
        if self.training:
            op4a = self.auxpool(X)
            op4a = self.aux_4a_convx1(op4a)
            op4a = self.activation(op4a)
            op4a = self.flatten(op4a)
            op4a = self.aux_4a_fc1(op4a)
            op4a = self.activation(op4a)
            op4a = self.aux_4a_fc2(op4a)
            op4a = self.aux_4a_dropout(op4a)

        X = self.inception4b(X)
        X = self.inception4c(X)
        X = self.inception4d(X)

        # aux classifier
        if self.training:
            op4d = self.auxpool(X)
            op4d = self.aux_4d_convx1(op4d)
            op4d = self.activation(op4d)
            op4d = self.flatten(op4d)
            op4d = self.aux_4d_fc1(op4d)
            op4d = self.activation(op4d)
            op4d = self.aux_4d_fc2(op4d)
            op4d = self.aux_4d_dropout(op4d)

        X = self.inception4e(X)
        X = self.mpool(X)
        X = self.inception5a(X)
        X = self.inception5b(X)
        X = self.fpool(X)
        X = self.dropout(X)
        X = self.flatten(X)
        X = self.fc(X)

        if self.training:
            return X, op4a, op4d
        return X
