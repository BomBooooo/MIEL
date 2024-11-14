import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp

class Linear_Backbone(nn.Module):
    def __init__(self, config):
        super(Linear_Backbone, self).__init__()
        self.block = nn.Linear(config.seq_len, config.pred_len)
    def forward(self, x):
        x = self.block(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.window_len = configs.window_len
        self.seq_len = configs.seq_len
        self.individual = configs.individual
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.affine = True
        self.eps = 1e-5

        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)

        if self.individual:
            self.Seasonal = nn.ModuleList()
            self.Trend = nn.ModuleList()

            for i in range(self.channels):
                # if self.backbone == 'Linear':
                self.Seasonal.append(
                    Linear_Backbone(configs)
                )
                self.Trend.append(
                    Linear_Backbone(configs)
                )
        else:
            self.Seasonal = Linear_Backbone(configs)
            self.Trend = Linear_Backbone(configs)

        # 不提前进行映射:self.seq_len * len(configs.window_len)
        self.reduce = nn.Sequential(
            nn.Linear(self.pred_len * len(configs.window_len), self.pred_len, bias=True),
            nn.Dropout(configs.dropout)
        )

        self.affine_weight = nn.Parameter(torch.ones(self.channels))
        self.affine_bias = nn.Parameter(torch.zeros(self.channels))


    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Seasonal[i](
                    seasonal_init[:, i, :].unsqueeze(1))[:,0,:]
                trend_output[:, i, :] = self.Trend[i](
                    trend_init[:, i, :].unsqueeze(1))[:,0,:]
        else:
            seasonal_output = self.Seasonal(seasonal_init)
            trend_output = self.Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def adaptive(self, enc_out):
        # 32*k,96,7—>32,k,96,7
        enc_out = torch.reshape(enc_out, (-1, len(self.window_len), enc_out.shape[-2], enc_out.shape[-1]))
        for j in range(len(self.window_len)):
            enc_out_tmp = enc_out[:, j, :, :]
            # enc_out_tmp = self.k_adaptive_layer[j](enc_out_tmp.permute(0, 2, 1)).permute(0, 2, 1)
            if j == 0:
                k_adaptive_enc_out = enc_out_tmp
            else:
                # 拼接方式：1.乘len(window_len)
                k_adaptive_enc_out = torch.concatenate([k_adaptive_enc_out,enc_out_tmp],axis=1)
        return k_adaptive_enc_out

    def forecast(self, x_enc):
        # instance_normalization
        # 按照[1,2]即按照k和seq_len进行标准化，相当于是把不同k的相同step等同看待
        # 之前的RevIn是按照时间步进行标准化，加入k后则是按照k*seq_len进行标准化—>因此最后按照这种方式融合效果会好？
        means = x_enc.mean([1, 2], keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=[1, 2], keepdim=True, unbiased=False) + self.eps)
        x_enc /= stdev
        if self.affine:
            x_enc = x_enc * self.affine_weight
            x_enc = x_enc + self.affine_bias

        # Encoder
        # 32,k,96,7—>32*k,96,7
        x_enc = torch.reshape(x_enc, (x_enc.shape[0] * x_enc.shape[1], x_enc.shape[2], x_enc.shape[3]))
        enc_out = self.encoder(x_enc)
        # k-CI
        enc_out = self.adaptive(enc_out)

        # Linear reduce
        dec_out = self.reduce(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # instance_normalization
        if self.affine:
            dec_out = dec_out - self.affine_bias
            dec_out = dec_out / (self.affine_weight + self.eps * self.eps)
        dec_out = dec_out * \
                  (stdev[:, 0, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
